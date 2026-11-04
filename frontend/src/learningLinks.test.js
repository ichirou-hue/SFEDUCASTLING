import { describe, expect, it } from "vitest";
import {
  getCourseTopicForPuzzle,
  getCourseTopicMeta,
} from "./learningLinks.js";

describe("course topic links", () => {
  it("возвращает метаданные известной темы", () => {
    expect(getCourseTopicMeta("tactics")).toEqual({
      slug: "tactics",
      title: "Тактические приёмы",
      icon: "♞",
    });
    expect(getCourseTopicMeta("unknown")).toBeNull();
  });

  it("сопоставляет тактические мотивы", () => {
    expect(getCourseTopicForPuzzle({ themes: ["fork"] })?.slug).toBe("tactics");
    expect(getCourseTopicForPuzzle({ themes: ["mateIn2"] })?.slug).toBe(
      "king-safety",
    );
    expect(getCourseTopicForPuzzle({ themes: ["promotion"] })?.slug).toBe(
      "endgames",
    );
  });

  it("возвращает null без тем", () => {
    expect(getCourseTopicForPuzzle({})).toBeNull();
    expect(getCourseTopicForPuzzle({ themes: ["unmapped"] })).toBeNull();
  });
});
