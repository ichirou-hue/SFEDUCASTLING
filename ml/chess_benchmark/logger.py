import os
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

# Исключаем локальный IPC-сокет W&B из проксирования
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1,localaddress,.local"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1,localaddress,.local"
os.environ["WANDB_INIT_TIMEOUT"] = "300"
os.environ["WANDB_HTTP_TIMEOUT"] = "300"
os.environ["WANDB_DISABLE_CODE"] = "true"

import wandb

console = Console()


class BenchmarkLogger:
    """Модуль генерации аналитических диаграмм и отправки структурированных отчетов в W&B."""

    def __init__(self, project: str, run_name: str, config: dict):
        if "/" in project:
            self.entity, self.project_name = project.split("/", 1)
        else:
            self.entity, self.project_name = None, project

        self.run_name = run_name
        self.config = config

    @staticmethod
    def _create_radar_chart(category_rows: list, model_name: str) -> plt.Figure:
        """Лепестковая диаграмма компетенций по 5 модулям (шкала 0-500)."""
        categories = [r[0] for r in category_rows]
        scores = [r[3] for r in category_rows]

        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        scores_closed = scores + [scores[0]]
        angles_closed = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#18181b')
        ax.set_facecolor('#18181b')

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles, categories, color='white', size=10, weight='bold')

        ax.set_rlabel_position(0)
        plt.yticks([100, 200, 300, 400, 500], ["100", "200", "300", "400", "500"], color="#a1a1aa", size=8)
        plt.ylim(0, 500)

        ax.plot(angles_closed, scores_closed, color='#38bdf8', linewidth=2.5, linestyle='solid')
        ax.fill(angles_closed, scores_closed, color='#38bdf8', alpha=0.35)

        ax.grid(color='#3f3f46', linestyle='--', linewidth=0.8)
        ax.spines['polar'].set_color('#3f3f46')
        plt.title(f"Chess Competencies: {model_name}", color='white', size=13, weight='bold', pad=20)
        plt.tight_layout()
        return fig

    @staticmethod
    def _create_donut_chart(total_score: float, max_score: float, accuracy: float) -> plt.Figure:
        """Круговой прогресс-бар (Donut Chart) для общего счета."""
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#18181b')
        ax.set_facecolor('#18181b')

        sizes = [total_score, max(0.0, max_score - total_score)]
        colors = ['#22c55e', '#27272a']

        ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.28, edgecolor='#18181b', linewidth=2)
        )

        ax.text(0, 0.1, f"{total_score:.1f}", ha='center', va='center', color='white', fontsize=22, weight='bold')
        ax.text(0, -0.15, f"/ {max_score:.0f} pts ({accuracy:.1f}%)", ha='center', va='center', color='#a1a1aa', fontsize=11)

        plt.title("Total Benchmark Score", color='white', size=13, weight='bold', pad=10)
        plt.tight_layout()
        return fig

    def log_results(self, summary_metrics: dict, category_rows: list, detailed_rows: list):
        """Рендерит графики и публикует структурированные данные в W&B."""
        console.print(f"[dim]⏳ Подготовка аналитики и отправка в W&B ({self.project_name})...[/]")

        try:
            settings = wandb.Settings(
                init_timeout=300.0,
                _disable_stats=True,
                _disable_meta=True,
                save_code=False,
                silent=True
            )

            run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=self.run_name,
                config=self.config,
                reinit="finish_previous",
                settings=settings
            )

            radar_fig = self._create_radar_chart(category_rows, self.config.get("model", "Model"))
            donut_fig = self._create_donut_chart(
                summary_metrics["Total_Score"],
                summary_metrics["Max_Score"],
                summary_metrics["Overall_Accuracy"]
            )

            cat_table = wandb.Table(
                columns=["Module", "Dataset", "Passed/Total", "Score (Max 500)", "Accuracy %"],
                data=category_rows
            )
            detail_table = wandb.Table(
                columns=["Module", "Status", "Question", "FEN", "Expected", "Model Output", "Validation Reason"],
                data=detailed_rows
            )

            payload = {
                "Visuals/Radar_Competencies": wandb.Image(radar_fig),
                "Visuals/Total_Score_Gauge": wandb.Image(donut_fig),
                "Benchmark_Summary/Category_Breakdown": cat_table,
                "Benchmark_Summary/Detailed_Predictions": detail_table,
                **summary_metrics
            }
            run.log(payload)

            for key, val in summary_metrics.items():
                run.summary[key] = val

            url = run.url
            plt.close(radar_fig)
            plt.close(donut_fig)
            run.finish(exit_code=0)

            console.print(f"[bold green]✔ Отчет успешно опубликован в W&B: [cyan]{self.run_name}[/][/]")
            console.print(f"[bold cyan]🔗 URL:[/] {url}")

        except Exception as e:
            console.print(f"[bold red]Ошибка W&B:[/] {e}")