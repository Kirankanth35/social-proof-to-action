from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, x, y, w, h, title, lines, fontsize=11):
    """
    Draw a rounded rectangle box with a title and bullet-style lines.
    (x, y) is lower-left corner in axis coordinates.
    """
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(box)

    text = title + "\n" + "\n".join([f"- {line}" for line in lines])
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        family="serif"
    )


def add_arrow(ax, start, end, label=None, dashed=False, label_offset=(0, 0), fontsize=11):
    """
    Draw an arrow between two points.
    """
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="->",
        mutation_scale=15,
        linewidth=1.5,
        linestyle="--" if dashed else "-",
        color="black"
    )
    ax.add_patch(arrow)

    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=fontsize, ha="center", va="center", family="serif")


def create_conceptual_model(output_png="figure1_conceptual_model.png", output_pdf="figure1_conceptual_model.pdf"):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main boxes
    add_box(
        ax, 0.08, 0.68, 0.26, 0.16,
        "Early Persuasive Signals",
        ["First-hour engagement", "Persuasive power index"]
    )

    add_box(
        ax, 0.40, 0.68, 0.26, 0.16,
        "Algorithmic Amplification",
        ["Reach", "Impressions", "Visibility / exposure"]
    )

    add_box(
        ax, 0.72, 0.68, 0.22, 0.16,
        "High-Effort Engagement Outcomes",
        ["Saves", "Profile visits", "Link clicks", "Engagement success"],
        fontsize=10
    )

    # Moderator box
    add_box(
        ax, 0.40, 0.28, 0.23, 0.14,
        "Influencer Authority",
        ["authority_log", "Follower-based credibility"],
        fontsize=10
    )

    # Boundary condition box
    add_box(
        ax, 0.72, 0.28, 0.22, 0.16,
        "Reach Saturation / High Visibility",
        ["log_reach", "very_high_reach", "Diminishing returns"],
        fontsize=10
    )

    # Solid arrows for main paths
    add_arrow(ax, (0.34, 0.76), (0.40, 0.76), label="H1", label_offset=(0, 0.03))
    add_arrow(ax, (0.66, 0.76), (0.72, 0.76), label="H2", label_offset=(0, 0.03))

    # Dashed moderator arrow to H1 path
    add_arrow(ax, (0.52, 0.42), (0.45, 0.72), label="H4, H5", dashed=True, label_offset=(-0.03, 0.02), fontsize=10)

    # Dashed boundary arrow to H2 path
    add_arrow(ax, (0.83, 0.44), (0.80, 0.70), label="H6, H7", dashed=True, label_offset=(0.04, 0.02), fontsize=10)

    # Direct dashed boundary-condition relation for RQ3
    add_arrow(ax, (0.22, 0.60), (0.82, 0.60), label="H8, H9", dashed=True, label_offset=(0, 0.03))

    # Title
    ax.text(
        0.5, 0.95,
        "Figure 1. Conceptual Research Model",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        family="serif"
    )

    # Caption
    caption = (
        "The model proposes that early persuasive signals increase algorithmic amplification, "
        "which in turn increases high-effort engagement outcomes. Influencer authority strengthens "
        "the translation of persuasive signals into amplification, while reach saturation and very "
        "high visibility act as boundary conditions that reduce the marginal effect of visibility "
        "and weaken the predictive role of persuasive power."
    )
    ax.text(
        0.5, 0.08,
        caption,
        ha="center",
        va="center",
        fontsize=10,
        family="serif",
        wrap=True
    )

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {Path(output_png).resolve()}")
    print(f"Saved: {Path(output_pdf).resolve()}")


if __name__ == "__main__":
    create_conceptual_model()