import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("../api.js", () => ({
  getAccessToken: vi.fn(),
  fetchLearningRecords: vi.fn(),
}));

import { fetchLearningRecords, getAccessToken } from "../api.js";
import UserRecordsWidget from "./UserRecordsWidget.jsx";

const records = {
  records: {
    weekly_best_opponent_elo: 1500,
    longest_win_streak: 3,
    best_puzzle_streak: 7,
  },
  activity: {
    starts_on: "2026-10-23",
    ends_on: "2026-11-05",
    puzzles: [{ date: "2026-10-23", correct: 2, incorrect: 1 }],
    games: [{ date: "2026-10-23", wins: 1, draws: 1, losses: 0 }],
  },
};

describe("UserRecordsWidget", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getAccessToken.mockReturnValue("token");
    fetchLearningRecords.mockResolvedValue(records);
  });

  it("показывает личные рекорды и график активности", async () => {
    render(<UserRecordsWidget />);

    expect(await screen.findByText("Личные рекорды")).toBeInTheDocument();
    expect(screen.getByText("1500")).toBeInTheDocument();
    expect(screen.getByText("ELO побеждённого бота")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
    expect(
      screen.getByRole("img", {
        name: "Активность: задачи за последние 14 дней",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("3 задачи")).toBeInTheDocument();
  });

  it("переключается на график партий", async () => {
    const user = userEvent.setup();
    render(<UserRecordsWidget />);

    await screen.findByText("Личные рекорды");
    await user.click(screen.getByRole("tab", { name: "Партии" }));

    expect(screen.getByRole("tab", { name: "Партии" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(
      screen.getByRole("img", {
        name: "Активность: партии за последние 14 дней",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("2 партии")).toBeInTheDocument();
  });

  it("обновляет данные после завершения партии", async () => {
    render(<UserRecordsWidget />);
    await screen.findByText("Личные рекорды");
    expect(fetchLearningRecords).toHaveBeenCalledTimes(1);

    window.dispatchEvent(new Event("sfedu-game-finished"));

    await waitFor(() => {
      expect(fetchLearningRecords).toHaveBeenCalledTimes(2);
    });
  });

  it("не показывает виджет гостю", () => {
    getAccessToken.mockReturnValue(null);
    const { container } = render(<UserRecordsWidget />);
    expect(container).toBeEmptyDOMElement();
    expect(fetchLearningRecords).not.toHaveBeenCalled();
  });

  it("показывает ошибку при сбое API", async () => {
    fetchLearningRecords.mockRejectedValueOnce(new Error("network"));
    render(<UserRecordsWidget />);

    expect(
      await screen.findByText(
        "Не удалось загрузить статистику. Обновите страницу или попробуйте позже.",
      ),
    ).toBeInTheDocument();
  });
});
