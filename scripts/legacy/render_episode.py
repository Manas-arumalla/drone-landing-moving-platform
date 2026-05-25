from __future__ import annotations

import argparse
import json
from pathlib import Path

from drone_landing.control import CascadedPIDController
from drone_landing.envs import LandingConfig, LandingEnv


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drone Landing Simulation</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #111418;
      color: #f3f6f8;
      display: grid;
      place-items: center;
      min-height: 100vh;
    }
    main { width: min(980px, calc(100vw - 32px)); }
    canvas {
      width: 100%;
      aspect-ratio: 16 / 9;
      display: block;
      background: #e9eef2;
      border: 1px solid #2d3640;
    }
    .bar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
      font-size: 14px;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 8px 12px;
      background: #26a269;
      color: white;
      cursor: pointer;
      font-weight: 700;
    }
    .stats { color: #cad3dc; }
  </style>
</head>
<body>
  <main>
    <div class="bar">
      <button id="play">Pause</button>
      <div class="stats" id="stats"></div>
    </div>
    <canvas id="scene" width="960" height="540"></canvas>
  </main>
  <script>
    const frames = __FRAMES__;
    const summary = __SUMMARY__;
    const canvas = document.getElementById("scene");
    const ctx = canvas.getContext("2d");
    const stats = document.getElementById("stats");
    const play = document.getElementById("play");
    let frame = 0;
    let running = true;
    const scale = 46;
    const origin = { x: canvas.width / 2, y: canvas.height * 0.78 };

    function sx(x) { return origin.x + x * scale; }
    function sy(z) { return origin.y - z * scale; }

    function draw() {
      const f = frames[frame];
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const sky = ctx.createLinearGradient(0, 0, 0, canvas.height);
      sky.addColorStop(0, "#d8e9f7");
      sky.addColorStop(1, "#f7fafb");
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = "#c5d0d8";
      ctx.lineWidth = 1;
      for (let x = -8; x <= 8; x += 1) {
        ctx.beginPath();
        ctx.moveTo(sx(x), sy(0));
        ctx.lineTo(sx(x), sy(5.2));
        ctx.stroke();
      }
      for (let z = 0; z <= 5; z += 1) {
        ctx.beginPath();
        ctx.moveTo(sx(-8), sy(z));
        ctx.lineTo(sx(8), sy(z));
        ctx.stroke();
      }

      ctx.fillStyle = "#65717c";
      ctx.fillRect(0, sy(0) + 8, canvas.width, canvas.height - sy(0));

      const platformX = sx(f.platform_x);
      const platformY = sy(0);
      ctx.fillStyle = "#16845f";
      ctx.fillRect(platformX - 38, platformY - 9, 76, 18);
      ctx.fillStyle = "#f7d154";
      ctx.beginPath();
      ctx.arc(platformX, platformY - 12, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = "#d04f4f";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= frame; i += 1) {
        const p = frames[i];
        const x = sx(p.drone_x);
        const y = sy(p.drone_z);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      const droneX = sx(f.drone_x);
      const droneY = sy(f.drone_z);
      ctx.strokeStyle = "#1d3557";
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(droneX - 24, droneY);
      ctx.lineTo(droneX + 24, droneY);
      ctx.moveTo(droneX, droneY - 24);
      ctx.lineTo(droneX, droneY + 24);
      ctx.stroke();
      ctx.fillStyle = "#284bba";
      ctx.beginPath();
      ctx.arc(droneX, droneY, 9, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#111418";
      ctx.font = "14px Arial";
      ctx.fillText(`t=${f.t.toFixed(2)}s  error=${f.error.toFixed(3)}m  z=${f.drone_z.toFixed(2)}m`, 18, 28);
      stats.textContent = `termination=${summary.termination} | success=${summary.success} | steps=${summary.steps} | return=${summary.return.toFixed(2)}`;

      if (running) frame = (frame + 1) % frames.length;
      requestAnimationFrame(draw);
    }

    play.addEventListener("click", () => {
      running = !running;
      play.textContent = running ? "Pause" : "Play";
    });

    draw();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "sinusoidal", "random_walk"], default="sinusoidal")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="runs/landing_episode.html")
    args = parser.parse_args()

    env = LandingEnv(LandingConfig(seed=args.seed, platform_mode=args.mode))
    policy = CascadedPIDController()
    obs = env.reset(seed=args.seed)
    frames = []
    total_reward = 0.0

    while True:
        frames.append(
            {
                "t": env.steps * env.config.dt,
                "drone_x": env.drone.x,
                "drone_y": env.drone.y,
                "drone_z": max(env.drone.z, 0.0),
                "platform_x": env.platform.x,
                "platform_y": env.platform.y,
                "error": env.horizontal_error(),
            }
        )
        result = env.step(policy.act(obs))
        obs = result.observation
        total_reward += result.reward
        if result.terminated or result.truncated:
            summary = {
                "termination": result.info["termination"],
                "success": bool(result.info["success"]),
                "steps": env.steps,
                "return": total_reward,
            }
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__FRAMES__", json.dumps(frames))
    html = html.replace("__SUMMARY__", json.dumps(summary))
    output.write_text(html, encoding="utf-8")
    print(f"wrote={output.resolve()}")
    print(f"success={summary['success']} termination={summary['termination']} steps={summary['steps']}")


if __name__ == "__main__":
    main()

