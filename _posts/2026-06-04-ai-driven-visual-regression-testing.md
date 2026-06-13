---
title: "AI-Driven Visual Regression Testing with Antigravity 2.0"
date: "2026-06-04 12:00:00 +0100"
categories: [Antigravity, Engineering]
tags: [Antigravity Engineering Series, Visual Testing, Browser Subagent, Hooks]
image:
  path: /assets/img/antigravity.png
  alt: "Antigravity Engineering Series by Amulya Bhatia"
---

> This article is part of the [Antigravity Engineering Series](https://iamulya.one/tags/antigravity-engineering-series). For any issues or if you'd like the pdf/epub version, contact me on [LinkedIn](https://www.linkedin.com/in/amulya-bhatia-01627a42/)
{: .prompt-info }

Your component library has 47 components. Each has 3–5 visual states. That's 180+ screenshots to verify after every design system change. Nobody does it. Instead, you merge the PR, someone notices the button padding is wrong in production three days later, and you open another PR to fix it.

Visual regression testing solves this, but traditional pixel-diff tools (Percy, Chromatic, BackstopJS) require explicit test setup, static baselines, and tuned thresholds. They catch pixel differences. They can't tell you whether a difference *matters*.

Antigravity's **Browser Subagent** changes the approach. Instead of comparing pixels, it *looks* at the UI through a sandboxed Chrome instance and reasons about what it sees. The subagent captures screenshots, records video, and can interact with elements — clicking buttons, filling forms, hovering for tooltips. Combined with **Sidecars** for scheduling and **Hooks** for safety, you get an autonomous visual QA pipeline that runs overnight and reports regressions with context, not just pixel counts.

---

## The Browser Subagent

The browser subagent is one of Antigravity's built-in subagent types. It operates a sandboxed Chrome browser with:

- **Screenshot capture**: Saves screenshots as artifacts
- **Video recording**: Records actions as WebM videos
- **UI actuation**: Can click, type, scroll, hover, and navigate
- **Chrome DevTools integration**: Natively connects to Chrome DevTools MCP
- **Isolated Chrome profile**: Runs in a completely separate Chrome profile — no access to your personal bookmarks, cookies, or sessions
- **URL security**: Governed by the `read_url` (viewing) and `execute_url` (actuation) permission actions

### Invoking the browser subagent

In the IDE, use the `/browser` command:

```
> /browser Navigate to http://localhost:3000/components and 
> take a screenshot of every component in its default state
```

The browser subagent spawns, opens Chrome, navigates to the URL, and begins capturing. You can watch its progress in the subagent panel. All screenshots and video recordings are saved as conversation artifacts.

---

## Building the Visual Regression Pipeline

### Step 1: The Design System Audit Skill

First, create a skill that teaches the agent how to evaluate UI components:

```
.agents/skills/visual-audit/
├── SKILL.md
└── resources/
    └── design-tokens.json
```

```markdown
---
name: visual-audit
description: >
  Performs visual regression testing of UI components by capturing
  screenshots and comparing them against design system specifications.
  Use when checking component rendering, spacing, colors, or 
  responsive behavior after CSS or component changes.
---

# Visual Audit Skill

## When to use this skill

- After any CSS, Tailwind config, or design token change
- After component library updates
- Before releasing a new design system version
- When verifying responsive behavior across breakpoints

## Audit workflow

1. Start the development server if not already running
2. Use the browser subagent to navigate to the component showcase
3. For each component category:
   a. Navigate to the component page
   b. Capture a screenshot of each visual state
   c. Compare against the design tokens in `resources/design-tokens.json`
   d. Check for:
      - Correct spacing (margins and padding per the token scale)
      - Color accuracy (compare against hex values in design tokens)
      - Typography (font family, weight, size per token definitions)
      - Interactive states (hover, focus, disabled, error)
      - Responsive behavior at 3 breakpoints (mobile: 375px, tablet: 768px, desktop: 1280px)
4. Report findings grouped by severity:
   - 🔴 Critical: Component is broken or visually incorrect
   - 🟡 Warning: Spacing or color is inconsistent with tokens
   - 🟢 Pass: Component matches design specifications

## Design token reference

Read `resources/design-tokens.json` for the authoritative design values.
When evaluating a component, compare the rendered output against these tokens.

## Screenshot naming

Save screenshots with this naming convention:
- `{component}-{state}-{breakpoint}.png`
- Example: `button-hover-desktop.png`, `card-default-mobile.png`
```

The design tokens resource:

```json
{
  "colors": {
    "primary": "#4285F4",
    "primary-hover": "#3367D6",
    "error": "#EA4335",
    "surface": "#FFFFFF",
    "on-surface": "#202124",
    "border": "#DADCE0"
  },
  "spacing": {
    "xs": "4px",
    "sm": "8px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px"
  },
  "typography": {
    "body": { "family": "Roboto", "size": "14px", "weight": 400 },
    "heading": { "family": "Google Sans", "size": "24px", "weight": 500 },
    "caption": { "family": "Roboto", "size": "12px", "weight": 400 }
  },
  "radius": {
    "sm": "4px",
    "md": "8px",
    "lg": "16px",
    "pill": "9999px"
  }
}
```

### Step 2: Permission Configuration

The browser subagent needs URL permissions. Configure them for the project:

**Allow list**:
```text
read_url(localhost)
execute_url(localhost)
command(npm run dev)
command(npm run storybook)
command(npx)
```

**Deny list**:
```text
execute_url(*)
read_url(*)
```

This pattern: **allow localhost, deny everything else.** The agent can view and interact with your local dev server but cannot navigate to external URLs.

### Step 3: Hooks for Regression Detection

Create a `PostToolUse` hook that fires after every browser screenshot to log results:

```json
{
  "visual-audit-logger": {
    "PostToolUse": [
      {
        "matcher": "browser_subagent",
        "hooks": [
          {
            "type": "command",
            "command": ".agents/hooks/log-visual-result.sh",
            "timeout": 10
          }
        ]
      }
    ]
  },
  "prevent-external-nav": {
    "PreToolUse": [
      {
        "matcher": "browser_subagent",
        "hooks": [
          {
            "type": "command",
            "command": ".agents/hooks/check-browser-url.sh",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

The logging hook:

```bash
#!/bin/bash
# .agents/hooks/log-visual-result.sh
# PostToolUse hook — logs browser subagent results to a JSONL audit trail

INPUT=$(cat)
STEP_IDX=$(echo "$INPUT" | jq -r '.stepIdx')
ERROR=$(echo "$INPUT" | jq -r '.error // empty')
CONV_ID=$(echo "$INPUT" | jq -r '.conversationId')
ARTIFACTS_DIR=$(echo "$INPUT" | jq -r '.artifactDirectoryPath')

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

ENTRY=$(jq -n \
  --arg ts "$TIMESTAMP" \
  --arg step "$STEP_IDX" \
  --arg conv "$CONV_ID" \
  --arg err "$ERROR" \
  --arg artifacts "$ARTIFACTS_DIR" \
  '{timestamp: $ts, step: ($step|tonumber), conversationId: $conv, error: $err, artifactsDir: $artifacts}')

echo "$ENTRY" >> .agents/visual-audit-log.jsonl

# Return empty response (PostToolUse doesn't support decisions)
echo '{}'
```

### Step 4: Scheduled Visual Regression with Sidecars

Create a sidecar that runs the visual audit nightly:

```
~/.gemini/config/sidecars/
└── visual-regression/
    └── sidecar.json
```

```json
{
  "description": "Nightly visual regression check — captures and evaluates all components",
  "builtin": "schedule",
  "args": [
    "0 22 * * 1-5",
    "agentapi",
    "new-conversation",
    "Run the visual-audit skill. Start the dev server with npm run dev. Then use the browser subagent to:\n\n1. Navigate to http://localhost:3000/storybook\n2. For each component in the sidebar:\n   a. Click the component to view it\n   b. Capture a screenshot of the default state\n   c. If the component has interactive states (hover, disabled, error), capture those too\n   d. Check the component against the design tokens in resources/design-tokens.json\n3. Capture responsive screenshots at 375px, 768px, and 1280px widths for the 5 most critical components (Button, Card, Modal, Input, Navigation)\n4. Compile a report:\n   - Total components checked\n   - Regressions found (with screenshots)\n   - Warnings (spacing or color inconsistencies)\n   - Components that pass\n5. If any critical regressions are found, create a GitHub issue with the screenshots attached."
  ],
  "restart_policy": "on-failure"
}
```

Enable it:

```json
{
  "sidecars": {
    "visual-regression": {
      "enabled": true,
      "projectId": "design-system"
    }
  }
}
```

### Step 5: The Stop Hook — Ensuring Complete Reports

Use a `Stop` hook to prevent the agent from stopping before the report is complete:

```bash
#!/bin/bash
# .agents/hooks/visual-report-check.sh
# Stop hook — ensures the visual audit agent produces a full report

INPUT=$(cat)
REASON=$(echo "$INPUT" | jq -r '.terminationReason')
FULLY_IDLE=$(echo "$INPUT" | jq -r '.fullyIdle')
ARTIFACTS_DIR=$(echo "$INPUT" | jq -r '.artifactDirectoryPath')

if [ "$REASON" = "model_stop" ] && [ "$FULLY_IDLE" = "true" ]; then
  # Check if screenshots were actually captured
  SCREENSHOT_COUNT=$(find "$ARTIFACTS_DIR" -name "*.png" -o -name "*.webp" 2>/dev/null | wc -l | tr -d ' ')
  
  if [ "$SCREENSHOT_COUNT" -lt 5 ]; then
    echo "{\"decision\": \"continue\", \"reason\": \"Only ${SCREENSHOT_COUNT} screenshots captured. The visual audit should cover at least the 5 critical components. Continue capturing screenshots.\"}"
    exit 0
  fi
fi

echo '{"decision": "allow"}'
```

---

## Interactive Visual Debugging (IDE)

The browser subagent's real power shows during interactive debugging. Instead of just capturing screenshots, you can have a conversation about what you see:

```
> /browser Navigate to localhost:3000/components/button
> and tell me if the hover state matches our design tokens

Agent spawns browser subagent, navigates, hovers, captures:

"The button hover state uses #2196F3 (Material Blue 500), but your 
design tokens specify #3367D6 for primary-hover. The color is 12% 
lighter than the specification.

Additionally, the hover transition is instant — no easing curve 
applied. Most design systems use a 150-200ms ease-in-out transition.

Recommendation: Update the CSS custom property --color-primary-hover 
from #2196F3 to #3367D6, and add transition: background-color 150ms 
ease-in-out to the button base class."
```

The browser subagent's video recording captures the interaction, saved as a WebM artifact:

```
> Show me the recording of the button hover test
```

Every browser session is automatically recorded. You can review recordings to see exactly what the agent saw and did.

---

## The Product Surface

| Capability | Product | Role |
|-----------|---------|------|
| Browser subagent, sidecar scheduling, hooks.json safety gates, permissions (`read_url`/`execute_url`), `agentapi` | **2.0** | The execution platform — schedule, run, and gate visual audits |
| `/browser` command, subagent panel, artifact review (screenshots + recordings), Chrome DevTools MCP, isolated Chrome profile | **IDE** | The visual testing surface — view, interact, and debug UI issues |

---

## What You've Built

An autonomous visual regression pipeline where:

1. **A sidecar runs nightly** via the `schedule` builtin and `agentapi` — triggering a fresh visual audit
2. **The browser subagent navigates your component library** — capturing screenshots and videos of every component state
3. **A SKILL.md teaches the agent your design system** — spacing tokens, color values, typography specs
4. **The agent reasons about what it sees** — not pixel diffs, but semantic evaluation ("the color is 12% lighter than the specification")
5. **Hooks gate the process** — logging every screenshot, preventing external navigation, ensuring complete reports
6. **Permissions scope access** — `read_url(localhost)` and `execute_url(localhost)` only, everything else denied
7. **The Stop hook prevents premature exits** — the agent must capture all critical components before stopping

The traditional visual regression tool says "47 pixels changed in button.png." This system says "the button hover color doesn't match your design token, and there's no transition easing applied." One gives you a diff. The other gives you a diagnosis.
