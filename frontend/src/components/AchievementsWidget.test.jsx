import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("../api.js", () => ({
  getAccessToken: vi.fn(),
  fetchAchievements: vi.fn(),
}));

import { fetchAchievements, getAccessToken } from "../api.js";
import AchievementsWidget from "./AchievementsWidget.jsx";

const achievements = [
  {
    code: "first_game",
    title: "Первая партия",
    description: "Завершите первую партию с ботом",
    reward: 10,
    icon: "♟",
    unlocked: true,
    awarded: 10,
    unlocked_at: "2026-11-05T10:00:00",
  },
  {
    code: "first_mate",
    title: "Первый мат",
    description: "Поставьте мат боту",
    reward: 25,
    icon: "♛",
    unlocked: false,
    awarded: 0,
    unlocked_at: null,
  },
  {
    code: "first_loss",
    title: "Первое поражение",
    description: "Проиграйте первую партию: это тоже опыт",
    reward: 5,
    icon: "♞",
    unlocked: true,
    awarded: 5,
    unlocked_at: "2026-11-05T11:00:00",
  },
];

describe("AchievementsWidget", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getAccessToken.mockReturnValue("token");
    fetchAchievements.mockResolvedValue(achievements);
  });

  it("показывает прогресс и награды", async () => {
    render(<AchievementsWidget />);

    expect(await screen.findByText("Достижения")).toBeInTheDocument();
    expect(screen.getByText("2 из 3")).toBeInTheDocument();
    expect(screen.getByText("Первая партия")).toBeInTheDocument();
    expect(screen.getAllByText("Получено")).toHaveLength(2);
    expect(screen.getByText("Не открыто")).toBeInTheDocument();
    expect(screen.getByText("+10 GigaCoin")).toBeInTheDocument();
    expect(screen.getByText("+25 GigaCoin")).toBeInTheDocument();
    expect(screen.getByText("+5 GigaCoin")).toBeInTheDocument();
    expect(
      screen.getByRole("article", { name: "Первый мат: не открыто" }),
    ).toBeInTheDocument();
  });

  it("не показывает виджет гостю", () => {
    getAccessToken.mockReturnValue(null);
    const { container } = render(<AchievementsWidget />);
    expect(container).toBeEmptyDOMElement();
    expect(fetchAchievements).not.toHaveBeenCalled();
  });

  it("подсказывает, как открыть первую награду", async () => {
    fetchAchievements.mockResolvedValueOnce([]);
    render(<AchievementsWidget />);

    expect(
      await screen.findByText(
        "Сыграйте первую партию, чтобы открыть первую награду.",
      ),
    ).toBeInTheDocument();
  });

  it("позволяет повторить загрузку после ошибки", async () => {
    const user = userEvent.setup();
    fetchAchievements
      .mockRejectedValueOnce(new Error("network"))
      .mockResolvedValueOnce(achievements);

    render(<AchievementsWidget />);
    expect(
      await screen.findByText("Не удалось загрузить достижения."),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Повторить" }));

    expect(await screen.findByText("Достижения")).toBeInTheDocument();
    expect(fetchAchievements).toHaveBeenCalledTimes(2);
  });

  it("обновляется после завершения партии", async () => {
    render(<AchievementsWidget />);
    await screen.findByText("Достижения");
    expect(fetchAchievements).toHaveBeenCalledTimes(1);

    window.dispatchEvent(new Event("sfedu-game-finished"));

    await waitFor(() => {
      expect(fetchAchievements).toHaveBeenCalledTimes(2);
    });
  });
});
