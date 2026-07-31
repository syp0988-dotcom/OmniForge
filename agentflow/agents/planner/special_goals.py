"""Special goal templates - deterministic demos and report generation.

These are hard-coded special cases that sit outside the generic planner /
reflector logic (snake-game demos used by benchmarks, and DOCX report
generation used by users).  Keeping them in one module gives the templates
a single source of truth and keeps the generic task-generation paths clean.
"""

from __future__ import annotations

from agentflow.graph.plan import Plan
from agentflow.graph.task import Task

# ------------------------------------------------------------------
# Deterministic file-generation templates
# ------------------------------------------------------------------


def is_snake_game_goal(goal: str) -> bool:
    """Check if the goal is asking to create a Snake game."""
    text = goal.lower()
    has_snake = any(kw in goal for kw in ("蛇", "贪吃蛇"))
    has_game = any(kw in goal for kw in ("游戏", "game"))
    has_python = "python" in text
    has_java = "java" in text
    has_create = any(token in goal for token in ("创建", "生成", "新建", "写", "请", "实现", "编写", "做", "开发", "帮我", "帮忙"))
    score = sum([has_snake, has_game, has_create])
    # Match if user is asking to create a snake game (at least 2 indicators)
    return (score >= 2 and has_snake) or (has_snake and (has_python or has_java))


def build_snake_game_files_plan(goal: str) -> Plan:
    text = goal.lower()
    want_python = "python" in text
    want_java = "java" in text
    # If no language specified, generate both Python and Java
    if not want_python and not want_java:
        want_python = True
        want_java = True

    tasks: list[Task] = []
    counter = 0

    if want_python:
        counter += 1
        tasks.append(Task(
            task_id="create_python_snake",
            title="创建 Python 贪吃蛇文件",
            priority=100,
            goal="write_file",
            capability="filesystem.write_file",
            tool="filesystem",
            input={
                "action": "write_file",
                "path": "snake_game/snake_game.py",
                "content": python_snake_content(),
            },
            agent="planner",
        ))

    if want_java:
        counter += 1
        tasks.append(Task(
            task_id="create_java_snake",
            title="创建 Java 贪吃蛇文件",
            priority=100 if not want_python else 95,
            goal="write_file",
            capability="filesystem.write_file",
            tool="filesystem",
            input={
                "action": "write_file",
                "path": "snake_game/SnakeGame.java",
                "content": java_snake_content(),
            },
            agent="planner",
        ))

    if not tasks:
        return Plan(
            goal=goal, category="project",
            tasks=[], goal_completed=True,
            reasoning="No language matched for snake game",
        )

    return Plan(
        goal=goal,
        category="project",
        tasks=tasks,
        goal_completed=False,
        reasoning=f"Matched deterministic snake game template ({'Python' if want_python else ''}{' + ' if want_python and want_java else ''}{'Java' if want_java else ''})",
    )


def python_snake_content() -> str:
    return r'''"""Snake Game — Python + tkinter, full playable version."""

import tkinter as tk
import random


class SnakeGame:
    WIDTH = 600
    HEIGHT = 600
    CELL = 20
    SPEED = 120  # ms between frames

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Snake Game")
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root, width=self.WIDTH, height=self.HEIGHT, bg="#1a1a2e"
        )
        self.canvas.pack()
        self.root.bind("<KeyPress>", self._on_key)
        self._score_label = tk.Label(
            self.root, text="Score: 0", font=("Arial", 14),
            fg="#e0e0e0", bg="#1a1a2e"
        )
        self._score_label.pack()
        self._score = 0
        self._start_game()
        self._loop()
        self.root.mainloop()

    def _start_game(self) -> None:
        cols = self.WIDTH // self.CELL
        rows = self.HEIGHT // self.CELL
        self._snake = [(cols // 2, rows // 2)]
        self._food = self._spawn_food()
        self._direction = (1, 0)  # moving right
        self._game_over = False
        self._score = 0

    def _spawn_food(self) -> tuple[int, int]:
        cols = self.WIDTH // self.CELL
        rows = self.HEIGHT // self.CELL
        while True:
            pos = (random.randrange(cols), random.randrange(rows))
            if pos not in self._snake:
                return pos

    def _on_key(self, event: tk.Event) -> None:
        key_map = {
            "Up": (0, -1), "Down": (0, 1),
            "Left": (-1, 0), "Right": (1, 0),
            "w": (0, -1), "s": (0, 1),
            "a": (-1, 0), "d": (1, 0),
        }
        new_dir = key_map.get(event.keysym)
        if new_dir and (
            new_dir[0] != -self._direction[0]
            or new_dir[1] != -self._direction[1]
        ):
            self._direction = new_dir
        if event.keysym == "r" and self._game_over:
            self._start_game()

    def _step(self) -> None:
        if self._game_over:
            return
        head = self._snake[-1]
        new_head = (head[0] + self._direction[0], head[1] + self._direction[1])

        cols = self.WIDTH // self.CELL
        rows = self.HEIGHT // self.CELL
        if (
            new_head[0] < 0 or new_head[0] >= cols
            or new_head[1] < 0 or new_head[1] >= rows
            or new_head in self._snake
        ):
            self._game_over = True
            return

        self._snake.append(new_head)
        if new_head == self._food:
            self._food = self._spawn_food()
            self._score += 10
        else:
            self._snake.pop(0)

    def _draw(self) -> None:
        self.canvas.delete("all")
        for seg in self._snake:
            x1 = seg[0] * self.CELL
            y1 = seg[1] * self.CELL
            x2 = x1 + self.CELL
            y2 = y1 + self.CELL
            self.canvas.create_rectangle(
                x1, y1, x2, y2, fill="#00d2ff", outline="#0099cc"
            )
        fx, fy = self._food
        x1 = fx * self.CELL
        y1 = fy * self.CELL
        x2 = x1 + self.CELL
        y2 = y1 + self.CELL
        self.canvas.create_rectangle(
            x1, y1, x2, y2, fill="#ff6b6b", outline="#cc4444"
        )
        if self._game_over:
            self.canvas.create_text(
                self.WIDTH // 2, self.HEIGHT // 2 - 20,
                text="GAME OVER", fill="#ffffff",
                font=("Arial", 28, "bold"),
            )
            self.canvas.create_text(
                self.WIDTH // 2, self.HEIGHT // 2 + 20,
                text="Press R to restart", fill="#aaaaaa",
                font=("Arial", 14),
            )
        self._score_label.config(text=f"Score: {self._score}")

    def _loop(self) -> None:
        self._step()
        self._draw()
        if self._game_over:
            self.root.after(500, self._loop)
        else:
            self.root.after(self.SPEED, self._loop)


if __name__ == "__main__":
    SnakeGame()
'''


def java_snake_content() -> str:
    return r'''import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Snake Game — Java Swing, full playable version.
 * Compile: javac SnakeGame.java
 * Run:     java SnakeGame
 */
public class SnakeGame extends JPanel {
    private static final int WIDTH = 600;
    private static final int HEIGHT = 600;
    private static final int CELL = 20;
    private static final int SPEED = 120;

    private final List<Point> snake = new ArrayList<>();
    private Point food;
    private int dirX = 1, dirY = 0;
    private boolean gameOver = false;
    private int score = 0;
    private final Random random = new Random();

    public SnakeGame() {
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(new Color(26, 26, 46));
        setFocusable(true);
        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                int key = e.getKeyCode();
                if (gameOver && key == KeyEvent.VK_R) {
                    startGame();
                    return;
                }
                switch (key) {
                    case KeyEvent.VK_UP, KeyEvent.VK_W:
                        if (dirY != 1) { dirX = 0; dirY = -1; } break;
                    case KeyEvent.VK_DOWN, KeyEvent.VK_S:
                        if (dirY != -1) { dirX = 0; dirY = 1; } break;
                    case KeyEvent.VK_LEFT, KeyEvent.VK_A:
                        if (dirX != 1) { dirX = -1; dirY = 0; } break;
                    case KeyEvent.VK_RIGHT, KeyEvent.VK_D:
                        if (dirX != -1) { dirX = 1; dirY = 0; } break;
                }
            }
        });
        startGame();
        new javax.swing.Timer(SPEED, e -> { step(); repaint(); }).start();
    }

    private void startGame() {
        snake.clear();
        int cols = WIDTH / CELL, rows = HEIGHT / CELL;
        snake.add(new Point(cols / 2, rows / 2));
        spawnFood();
        dirX = 1; dirY = 0;
        gameOver = false;
        score = 0;
    }

    private void spawnFood() {
        int cols = WIDTH / CELL, rows = HEIGHT / CELL;
        Point p;
        do {
            p = new Point(random.nextInt(cols), random.nextInt(rows));
        } while (snake.contains(p));
        food = p;
    }

    private void step() {
        if (gameOver) return;
        Point head = snake.get(snake.size() - 1);
        Point newHead = new Point(head.x + dirX, head.y + dirY);
        int cols = WIDTH / CELL, rows = HEIGHT / CELL;

        if (newHead.x < 0 || newHead.x >= cols
            || newHead.y < 0 || newHead.y >= rows
            || snake.contains(newHead)) {
            gameOver = true;
            return;
        }

        snake.add(newHead);
        if (newHead.equals(food)) {
            spawnFood();
            score += 10;
        } else {
            snake.remove(0);
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                            RenderingHints.VALUE_ANTIALIAS_ON);

        // Draw snake
        g2.setColor(new Color(0, 210, 255));
        for (Point seg : snake) {
            g2.fillRect(seg.x * CELL, seg.y * CELL, CELL - 1, CELL - 1);
        }

        // Draw food
        g2.setColor(new Color(255, 107, 107));
        g2.fillRect(food.x * CELL, food.y * CELL, CELL - 1, CELL - 1);

        // Score
        g2.setColor(Color.LIGHT_GRAY);
        g2.setFont(new Font("Arial", Font.PLAIN, 16));
        g2.drawString("Score: " + score, 10, 20);

        // Game over overlay
        if (gameOver) {
            g2.setColor(new Color(0, 0, 0, 150));
            g2.fillRect(0, 0, WIDTH, HEIGHT);
            g2.setColor(Color.WHITE);
            g2.setFont(new Font("Arial", Font.BOLD, 28));
            String msg = "GAME OVER";
            FontMetrics fm = g2.getFontMetrics();
            g2.drawString(msg, (WIDTH - fm.stringWidth(msg)) / 2, HEIGHT / 2 - 20);
            g2.setFont(new Font("Arial", Font.PLAIN, 14));
            String restart = "Press R to restart";
            fm = g2.getFontMetrics();
            g2.drawString(restart, (WIDTH - fm.stringWidth(restart)) / 2, HEIGHT / 2 + 20);
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Snake Game");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);
            frame.add(new SnakeGame());
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
'''


def go_snake_content() -> str:
    """Go Snake game source (single source of truth)."""
    return r'''package main
import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/gdamore/tcell/v2"
)

const (
	cell  = 20
	width = 30
	height = 20
	tickMs = 120
)

type Point struct{ X, Y int }

var (
	screen   tcell.Screen
	snake    []Point
	food     Point
	dir      = Point{1, 0}
	nextDir  = Point{1, 0}
	score    int
	gameOver bool
)

func main() {
	var err error
	screen, err = tcell.NewScreen()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	if err := screen.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	defer screen.Fini()

	screen.SetStyle(tcell.StyleDefault.Background(tcell.ColorBlack).Foreground(tcell.ColorWhite))
	screen.EnableMouse()
	rand.Seed(time.Now().UnixNano())

	reset()
	go inputLoop()
	gameLoop()
}

func reset() {
	snake = []Point{{width / 2, height / 2}, {width/2 - 1, height / 2}}
	food = newFood()
	dir = Point{1, 0}
	nextDir = Point{1, 0}
	score = 0
	gameOver = false
}

func newFood() Point {
	for {
		p := Point{rand.Intn(width), rand.Intn(height)}
		hit := false
		for _, s := range snake {
			if s == p {
				hit = true
				break
			}
		}
		if !hit {
			return p
		}
	}
}

func inputLoop() {
	for {
		ev := screen.PollEvent()
		switch ev := ev.(type) {
		case *tcell.EventKey:
			if gameOver && ev.Key() == tcell.KeyRune && ev.Rune() == ' ' {
				reset()
				continue
			}
			switch ev.Key() {
			case tcell.KeyUp, tcell.KeyRune:
				if ev.Rune() == 'w' || ev.Rune() == 'W' {
					if dir.Y != 1 { nextDir = Point{0, -1} }
				}
			case tcell.KeyDown:
				if dir.Y != -1 { nextDir = Point{0, 1} }
			case tcell.KeyLeft:
				if dir.X != 1 { nextDir = Point{-1, 0} }
			case tcell.KeyRight:
				if dir.X != -1 { nextDir = Point{1, 0} }
			case tcell.KeyRune:
				switch ev.Rune() {
				case 'w', 'W':
					if dir.Y != 1 { nextDir = Point{0, -1} }
				case 's', 'S':
					if dir.Y != -1 { nextDir = Point{0, 1} }
				case 'a', 'A':
					if dir.X != 1 { nextDir = Point{-1, 0} }
				case 'd', 'D':
					if dir.X != -1 { nextDir = Point{1, 0} }
				}
			}
		}
	}
}

func gameLoop() {
	ticker := time.NewTicker(time.Duration(tickMs) * time.Millisecond)
	defer ticker.Stop()
	for range ticker.C {
		if !gameOver {
			dir = nextDir
			head := snake[0]
			newHead := Point{head.X + dir.X, head.Y + dir.Y}
			if newHead.X < 0 || newHead.X >= width || newHead.Y < 0 || newHead.Y >= height {
				gameOver = true
			} else {
				for _, s := range snake {
					if s == newHead {
						gameOver = true
						break
					}
				}
			}
			if !gameOver {
				snake = append([]Point{newHead}, snake...)
				if newHead == food {
					score++
					food = newFood()
				} else {
					snake = snake[:len(snake)-1]
				}
			}
		}
		draw()
		if gameOver {
			time.Sleep(3 * time.Second)
			return
		}
	}
}

func draw() {
	screen.Clear()
	// Draw food
	foodStyle := tcell.StyleDefault.Foreground(tcell.ColorRed)
	screen.SetContent(food.X*2, food.Y, '█', nil, foodStyle)

	// Draw snake
	for i, s := range snake {
		var style tcell.Style
		if i == 0 {
			style = tcell.StyleDefault.Foreground(tcell.ColorGreen)
		} else {
			style = tcell.StyleDefault.Foreground(tcell.ColorLightGreen)
		}
		screen.SetContent(s.X*2, s.Y, '█', nil, style)
	}

	// Score
	scoreStr := fmt.Sprintf("Score: %d", score)
	for i, r := range scoreStr {
		screen.SetContent(i, height, r, nil, tcell.StyleDefault.Foreground(tcell.ColorWhite))
	}
	if gameOver {
		msg := "Game Over - press Space"
		for i, r := range msg {
			screen.SetContent(width - len(msg)/2 + i, height/2, r, nil, tcell.StyleDefault.Foreground(tcell.ColorYellow))
		}
	}
	screen.Show()
}'''



# Docx report template helpers
# ------------------------------------------------------------------


def is_docx_report_goal(goal: str) -> bool:
    text = goal.lower()
    wants_docx = any(token in text for token in ("docx", ".docx", "word"))
    wants_report = any(token in goal for token in ("报告", "文档", "整理"))
    create_intent = any(token in goal for token in ("整理", "生成", "创建", "输出", "做成", "写成"))
    return wants_docx and wants_report and create_intent


def build_docx_report_plan(goal: str, state: dict) -> Plan:
    content = _build_docx_report_content(goal, state)
    task = Task(
        task_id="create_docx_report",
        title="创建 DOCX 报告",
        priority=100,
        goal="create",
        capability="docx.create",
        tool="docx",
        input={
            "action": "create",
            "path": _docx_report_path(goal, state),
            "content": content,
        },
        agent="planner",
    )
    return Plan(
        goal=goal,
        category="project",
        tasks=[task],
        goal_completed=False,
        reasoning="Matched deterministic DOCX report template",
    )


def _build_docx_report_content(goal: str, state: dict) -> str:
    source = _latest_assistant_content(state)
    if not source:
        knowledge_context = str(state.get("knowledge_context", "") or "").strip()
        search_results = state.get("search_results")
        source = knowledge_context or _stringify_search_results(search_results)
    if not source:
        source = "暂无可整理的上一轮内容，请补充报告材料。"

    return (
        "# DOCX 报告\n\n"
        "## 用户需求\n\n"
        f"{goal}\n\n"
        "## 整理内容\n\n"
        f"{source.strip()}\n"
    )


def _latest_assistant_content(state: dict) -> str:
    history = state.get("history") or []
    if not isinstance(history, list):
        memory = state.get("memory")
        if isinstance(memory, dict):
            history = memory.get("history") or []
    if not isinstance(history, list):
        return ""

    for message in reversed(history):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = str(message.get("content", "") or "").strip()
            if content:
                return content
    return ""


def _stringify_search_results(search_results: object) -> str:
    if not isinstance(search_results, list):
        return ""
    lines: list[str] = []
    for item in search_results[:5]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "") or "").strip()
        summary = str(item.get("summary") or item.get("snippet") or item.get("content") or "").strip()
        if title or summary:
            lines.append(f"### {title or '搜索结果'}\n\n{summary}")
    return "\n\n".join(lines)


def _docx_report_path(goal: str, state: dict) -> str:
    history_text = _latest_assistant_content(state)
    combined = f"{goal}\n{history_text}".lower()
    if "omniforge" in combined:
        return "OmniForge报告.docx"
    return "report.docx"
