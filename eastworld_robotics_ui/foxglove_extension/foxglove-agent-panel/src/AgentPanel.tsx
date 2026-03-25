import { Immutable, PanelExtensionContext, RenderState, SettingsTreeAction } from "@foxglove/extension";
import { bannerDataUrl } from "./assets/banner";
import {
  CSSProperties,
  ReactElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";

// ── Types ──────────────────────────────────────────────────────────────────────

interface ChatMessage {
  role: "user" | "assistant" | "thought";
  content: string;
  timestamp: number;
  communicationId?: string;
  sessionHash?: string;
}

interface ToolEvent {
  id: string; // tool name + start timestamp
  tool: string;
  status: "running" | "done" | "error";
  args?: Record<string, unknown>;
  result?: unknown;
  elapsed_s?: number | null;
  error?: string;
}

/** ROS 2 std_msgs/msg/String */
interface RosStringMsg {
  data: string;
}

type ColorScheme = "dark" | "light";
type RobotStatus = "available" | "busy" | "unknown";
type AgentStatus = "idle" | "thinking" | "executing" | "error" | "unknown";

/** Shape of an incoming rai_interfaces/msg/HRIMessage over rosbridge. */
interface RosbridgeHRIPayload {
  op?: string;
  topic?: string;
  msg?: {
    header?: { stamp?: { sec?: number; nanosec?: number }; frame_id?: string };
    text?: string;
    images?: unknown[];
    audios?: unknown[];
    communication_id?: string;
    seq_no?: number;
    seq_end?: boolean;
  };
}

interface RosbridgeStringPayload {
  op?: string;
  topic?: string;
  msg?: { data?: string };
}

// ── Constants ─────────────────────────────────────────────────────────────────

const RAI_MSG_TYPE = "rai_interfaces/msg/HRIMessage";
const STRING_MSG_TYPE = "std_msgs/msg/String";
const DEFAULT_INPUT_TOPIC = "/from_human";
const DEFAULT_OUTPUT_TOPIC = "/to_human";
const DEFAULT_ROSBRIDGE_URL = "ws://localhost:9090";
const DEFAULT_TASK_SERVER_URL = "http://localhost:8090";
const DEFAULT_MODEL = "qwen3-vl:8b";
const TASK_POLL_INTERVAL_MS = 2000;
const MODEL_OPTIONS = [
  { label: "Qwen3-VL 8B (fast)", value: "qwen3-vl:8b" },
  { label: "Qwen3-VL 32B Thinking (reasoning)", value: "qwen3-vl:32b-thinking" },
];

type Embodiment = "scout-mini-rover" | "x500-drone" | "g1-humanoid";
const DEFAULT_EMBODIMENT: Embodiment = "scout-mini-rover";

function generateSessionHash(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(8)))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// ── Theme ──────────────────────────────────────────────────────────────────────

interface ThemeColors {
  bg: string;
  bgSurface: string;
  text: string;
  textMuted: string;
  textFaint: string;
  border: string;
  accent: string;
  accentText: string;
  assistantBg: string;
  assistantBorder: string;
  thoughtBg: string;
  thoughtBorder: string;
  thoughtText: string;
}

const THEMES: Record<ColorScheme, ThemeColors> = {
  dark: {
    bg: "#1e1e1e",
    bgSurface: "#2d2d2d",
    text: "#e0e0e0",
    textMuted: "#888",
    textFaint: "#666",
    border: "#333",
    accent: "#4a9eff",
    accentText: "#fff",
    assistantBg: "#2d2d2d",
    assistantBorder: "#444",
    thoughtBg: "#1a2a1a",
    thoughtBorder: "#2d5a2d",
    thoughtText: "#8fbc8f",
  },
  light: {
    bg: "#ffffff",
    bgSurface: "#f5f5f5",
    text: "#1a1a1a",
    textMuted: "#666",
    textFaint: "#999",
    border: "#e0e0e0",
    accent: "#1976d2",
    accentText: "#fff",
    assistantBg: "#f0f0f0",
    assistantBorder: "#ddd",
    thoughtBg: "#f0f8f0",
    thoughtBorder: "#a5d6a7",
    thoughtText: "#2e7d32",
  },
};

function buildStyles(t: ThemeColors) {
  return {
    container: {
      display: "flex",
      flexDirection: "column" as const,
      width: "100%",
      height: "100%",
      backgroundColor: t.bg,
      color: t.text,
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      fontSize: "13px",
      overflow: "hidden",
    } satisfies CSSProperties,
    header: {
      padding: "10px 16px",
      borderBottom: `1px solid ${t.border}`,
      fontWeight: 600,
      fontSize: "14px",
      color: t.text,
      flexShrink: 0,
      display: "flex",
      alignItems: "center",
      gap: "6px",
    } satisfies CSSProperties,
    section: {
      borderBottom: `1px solid ${t.border}`,
      flexShrink: 0,
    } satisfies CSSProperties,
    sectionHeader: {
      padding: "6px 16px",
      fontSize: "11px",
      fontWeight: 600,
      color: t.textMuted,
      cursor: "pointer",
      userSelect: "none" as const,
      display: "flex",
      alignItems: "center",
      gap: "4px",
    } satisfies CSSProperties,
    sectionBody: {
      padding: "4px 16px 8px",
      fontSize: "11px",
      color: t.textMuted,
      fontFamily: "monospace",
      maxHeight: "120px",
      overflowY: "auto" as const,
      whiteSpace: "pre-wrap" as const,
      wordBreak: "break-word" as const,
    } satisfies CSSProperties,
    messagesContainer: {
      flex: 1,
      overflowY: "auto" as const,
      padding: "12px 16px",
      display: "flex",
      flexDirection: "column" as const,
      gap: "8px",
    } satisfies CSSProperties,
    userBubble: {
      alignSelf: "flex-end" as const,
      backgroundColor: t.accent,
      color: t.accentText,
      padding: "8px 12px",
      borderRadius: "12px 12px 2px 12px",
      maxWidth: "80%",
      wordBreak: "break-word" as const,
      lineHeight: "1.4",
    } satisfies CSSProperties,
    assistantBubble: {
      alignSelf: "flex-start" as const,
      backgroundColor: t.assistantBg,
      color: t.text,
      padding: "8px 12px",
      borderRadius: "12px 12px 12px 2px",
      maxWidth: "80%",
      wordBreak: "break-word" as const,
      lineHeight: "1.4",
      border: `1px solid ${t.assistantBorder}`,
    } satisfies CSSProperties,
    thoughtBubble: {
      alignSelf: "flex-start" as const,
      backgroundColor: t.thoughtBg,
      color: t.thoughtText,
      padding: "6px 10px",
      borderRadius: "8px",
      maxWidth: "90%",
      wordBreak: "break-word" as const,
      lineHeight: "1.3",
      border: `1px solid ${t.thoughtBorder}`,
      fontSize: "12px",
      fontStyle: "italic" as const,
    } satisfies CSSProperties,
    roleLabel: {
      fontSize: "10px",
      color: t.textMuted,
      marginBottom: "2px",
      textTransform: "uppercase" as const,
      letterSpacing: "0.5px",
    } satisfies CSSProperties,
    inputContainer: {
      padding: "12px 16px",
      borderTop: `1px solid ${t.border}`,
      display: "flex",
      gap: "8px",
      flexShrink: 0,
    } satisfies CSSProperties,
    input: {
      flex: 1,
      padding: "8px 12px",
      backgroundColor: t.bgSurface,
      color: t.text,
      border: `1px solid ${t.assistantBorder}`,
      borderRadius: "8px",
      outline: "none",
      fontSize: "13px",
      fontFamily: "inherit",
    } satisfies CSSProperties,
    sendButton: {
      padding: "8px 16px",
      backgroundColor: t.accent,
      color: t.accentText,
      border: "none",
      borderRadius: "8px",
      cursor: "pointer",
      fontWeight: 600,
      fontSize: "13px",
      flexShrink: 0,
    } satisfies CSSProperties,
    emptyState: {
      display: "flex",
      flexDirection: "column" as const,
      alignItems: "center" as const,
      justifyContent: "center" as const,
      flex: 1,
      color: t.textFaint,
      gap: "8px",
      padding: "24px",
      textAlign: "center" as const,
    } satisfies CSSProperties,
  };
}

function agentStatusDotStyle(agentStatus: AgentStatus, connected: boolean): CSSProperties {
  let color = "#9e9e9e"; // unknown / disconnected
  let animation: string | undefined;
  if (connected) {
    switch (agentStatus) {
      case "idle":
        color = "#4caf50";
        break;
      case "thinking":
        color = "#4a9eff";
        animation = "pulse 1s infinite";
        break;
      case "executing":
        color = "#ff9800";
        animation = "pulse 0.7s infinite";
        break;
      case "error":
        color = "#f44336";
        break;
    }
  }
  return {
    display: "inline-block",
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    backgroundColor: color,
    flexShrink: 0,
    ...(animation ? { animation } : {}),
  };
}

// ── Helpers ─────────────────────────────────────────────────────────────────────

let _seqCounter = 0;

function buildHRIMessage(text: string, communicationId: string) {
  const now = Date.now();
  const sec = Math.floor(now / 1000);
  const nanosec = (now % 1000) * 1_000_000;
  _seqCounter += 1;
  return {
    header: {
      stamp: { sec, nanosec },
      frame_id: "",
    },
    text,
    images: [],
    audios: [],
    communication_id: communicationId,
    seq_no: _seqCounter,
    seq_end: true,
  };
}

// ── Topics ──────────────────────────────────────────────────────────────────────

const COMMAND_TOPIC = "/agent/command";
const RESPONSE_TOPIC = "/agent/response";
const THOUGHT_TOPIC = "/agent/thought";
const STRING_SCHEMA = "std_msgs/msg/String";

// ── Component ──────────────────────────────────────────────────────────────────

function AgentPanel({ context }: { context: PanelExtensionContext }): ReactElement {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [rosbridgeUrl, setRosbridgeUrl] = useState(DEFAULT_ROSBRIDGE_URL);
  const [taskServerUrl, setTaskServerUrl] = useState(DEFAULT_TASK_SERVER_URL);
  const [inputTopic, setInputTopic] = useState(DEFAULT_INPUT_TOPIC);
  const [outputTopic, setOutputTopic] = useState(DEFAULT_OUTPUT_TOPIC);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [embodiment, setEmbodiment] = useState<Embodiment>(DEFAULT_EMBODIMENT);
  const [connected, setConnected] = useState(false);
  const [renderDone, setRenderDone] = useState<(() => void) | undefined>();
  const [colorScheme, setColorScheme] = useState<ColorScheme>("dark");

  // Task server state
  const [robotStatus, setRobotStatus] = useState<RobotStatus>("unknown");
  const [agentStatus, setAgentStatus] = useState<AgentStatus>("unknown");
  const [currentTaskId, setCurrentTaskId] = useState<number | null>(null);
  const [reasoning, setReasoning] = useState("");
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [reasoningOpen, setReasoningOpen] = useState(false);
  const [toolsOpen, setToolsOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reasoningRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const outputTopicRef = useRef(outputTopic);
  outputTopicRef.current = outputTopic;
  const inputTopicRef = useRef(inputTopic);
  inputTopicRef.current = inputTopic;
  const taskServerUrlRef = useRef(taskServerUrl);
  taskServerUrlRef.current = taskServerUrl;
  const currentTaskIdRef = useRef(currentTaskId);
  currentTaskIdRef.current = currentTaskId;
  const inputRef = useRef<HTMLInputElement>(null);

  const styles = useMemo(() => buildStyles(THEMES[colorScheme]), [colorScheme]);

  // ── Inject pulse keyframes ─────────────────────────────────────────────────
  useEffect(() => {
    const id = "agent-panel-keyframes";
    if (!document.getElementById(id)) {
      const style = document.createElement("style");
      style.id = id;
      style.textContent = `@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.2; } }`;
      document.head.appendChild(style);
    }
  }, []);

  // ── Foxglove topic setup & render handler ─────────────────────────────────

  useLayoutEffect(() => {
    // Subscribe to agent responses and thoughts via Foxglove's native topic API.
    // Messages arrive in renderState.currentFrame during onRender.
    context.subscribe([{ topic: RESPONSE_TOPIC }, { topic: THOUGHT_TOPIC }]);

    // Advertise intent to publish commands (for rosbridge, no extra options needed).
    context.advertise?.(COMMAND_TOPIC, STRING_SCHEMA);

    context.onRender = (renderState: Immutable<RenderState>, done: () => void) => {
      // ── Theme ──
      if (renderState.colorScheme) {
        setColorScheme(renderState.colorScheme);
      }

      // ── Connection status ──
      // If Foxglove has topics from the data source, we're connected.
      if (renderState.topics != undefined) {
        setConnected(renderState.topics.length > 0);
      }

      // ── Incoming messages ──
      // currentFrame contains all messages received since the last render.
      if (renderState.currentFrame) {
        for (const event of renderState.currentFrame) {
          if (event.topic === RESPONSE_TOPIC) {
            const rosMsg = event.message as RosStringMsg;
            if (rosMsg.data) {
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: rosMsg.data, timestamp: Date.now() },
              ]);
            }
          } else if (event.topic === THOUGHT_TOPIC) {
            const rosMsg = event.message as RosStringMsg;
            if (rosMsg.data) {
              setMessages((prev) => [
                ...prev,
                { role: "thought", content: rosMsg.data, timestamp: Date.now() },
              ]);
            }
          }
        }
      }

      setRenderDone(() => done);
    };

    context.watch("currentFrame");
    context.watch("colorScheme");
    context.watch("topics");

    return () => {
      context.unsubscribeAll();
      context.unadvertise?.(COMMAND_TOPIC);
    };
  }, [context]);

  useEffect(() => {
    renderDone?.();
  }, [renderDone]);

  // ── Settings ───────────────────────────────────────────────────────────────

  const settingsActionHandler = useCallback((action: SettingsTreeAction) => {
    if (action.action === "update") {
      const { path, value } = action.payload;
      if (typeof value !== "string" || value.length === 0) {
        return;
      }
      switch (path[1]) {
        case "rosbridgeUrl":
          setRosbridgeUrl(value);
          break;
        case "taskServerUrl":
          setTaskServerUrl(value);
          break;
        case "inputTopic":
          setInputTopic(value);
          break;
        case "outputTopic":
          setOutputTopic(value);
          break;
        case "model":
          setModel(value);
          break;
        case "embodiment":
          setEmbodiment(value as Embodiment);
          break;
      }
    }
  }, []);

  useEffect(() => {
    context.updatePanelSettingsEditor({
      actionHandler: settingsActionHandler,
      nodes: {
        connection: {
          label: "Connection",
          fields: {
            rosbridgeUrl: {
              label: "Rosbridge URL",
              input: "string",
              value: rosbridgeUrl,
            },
          },
        },
        taskServer: {
          label: "Task Server",
          fields: {
            taskServerUrl: {
              label: "Task Server URL",
              input: "string",
              value: taskServerUrl,
            },
          },
        },
        topics: {
          label: "RAI Topics",
          fields: {
            inputTopic: {
              label: "Send to (input topic)",
              input: "string",
              value: inputTopic,
            },
            outputTopic: {
              label: "Listen on (output topic)",
              input: "string",
              value: outputTopic,
            },
          },
        },
        inference: {
          label: "Model",
          fields: {
            model: {
              label: "Model",
              input: "select",
              value: model,
              options: MODEL_OPTIONS,
            },
          },
        },
      },
    });
  }, [context, rosbridgeUrl, taskServerUrl, inputTopic, outputTopic, model, settingsActionHandler]);

  // ── Rosbridge WebSocket ────────────────────────────────────────────────────

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      try {
        ws = new WebSocket(rosbridgeUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);

          // Subscribe to HRIMessage output (agent replies)
          ws.send(JSON.stringify({ op: "subscribe", topic: outputTopic, type: RAI_MSG_TYPE }));

          // Subscribe to incoming tasks (for curl / external submissions)
          ws.send(JSON.stringify({ op: "subscribe", topic: inputTopic, type: RAI_MSG_TYPE }));

          // Subscribe to agent telemetry topics
          ws.send(JSON.stringify({ op: "subscribe", topic: "/agent/status", type: STRING_MSG_TYPE }));
          ws.send(JSON.stringify({ op: "subscribe", topic: "/agent/reasoning", type: STRING_MSG_TYPE }));
          ws.send(JSON.stringify({ op: "subscribe", topic: "/agent/events", type: STRING_MSG_TYPE }));

          // Advertise input topic (for direct rosbridge fallback only)
          ws.send(JSON.stringify({ op: "advertise", topic: inputTopic, type: RAI_MSG_TYPE }));
        };

        ws.onmessage = (event: MessageEvent) => {
          try {
            const data = JSON.parse(event.data as string) as RosbridgeHRIPayload & RosbridgeStringPayload;
            if (data.op !== "publish") return;

            if (data.topic === inputTopicRef.current && data.msg && "text" in data.msg && data.msg.text) {
              const userMsg: ChatMessage = {
                role: "user",
                content: data.msg.text,
                timestamp: Date.now(),
                communicationId: (data.msg as RosbridgeHRIPayload["msg"])?.communication_id,
              };
              setMessages((prev) => [...prev, userMsg]);
              setReasoning("");
              setToolEvents([]);
              setReasoningOpen(true);
            } else if (data.topic === outputTopicRef.current && data.msg && "text" in data.msg && data.msg.text) {
              const newMsg: ChatMessage = {
                role: "assistant",
                content: data.msg.text,
                timestamp: Date.now(),
                communicationId: (data.msg as RosbridgeHRIPayload["msg"])?.communication_id,
              };
              setMessages((prev) => [...prev, newMsg]);
            } else if (data.topic === "/agent/status" && data.msg?.data != null) {
              const raw = data.msg.data;
              if (raw.startsWith("executing:")) {
                setAgentStatus("executing");
              } else if (raw === "thinking") {
                setAgentStatus("thinking");
                setReasoning("");
                setToolEvents([]);
                setReasoningOpen(true);
              } else if (raw === "idle") {
                setAgentStatus("idle");
              } else if (raw === "error") {
                setAgentStatus("error");
              }
            } else if (data.topic === "/agent/reasoning" && data.msg?.data) {
              setReasoning((prev) => prev + data.msg!.data);
            } else if (data.topic === "/agent/events" && data.msg?.data) {
              try {
                const ev = JSON.parse(data.msg.data) as {
                  event: string;
                  tool?: string;
                  args?: Record<string, unknown>;
                  result?: unknown;
                  elapsed_s?: number | null;
                  error?: string;
                };
                if (ev.event === "start" && ev.tool) {
                  const id = `${ev.tool}-${Date.now()}`;
                  setToolEvents((prev) => [
                    ...prev,
                    { id, tool: ev.tool!, status: "running", args: ev.args },
                  ]);
                  setToolsOpen(true);
                } else if (ev.event === "end") {
                  setToolEvents((prev) => {
                    const idx = [...prev].reverse().findIndex((t) => t.status === "running");
                    if (idx === -1) return prev;
                    const realIdx = prev.length - 1 - idx;
                    const updated = [...prev];
                    updated[realIdx] = {
                      ...updated[realIdx]!,
                      status: "done",
                      result: ev.result,
                      elapsed_s: ev.elapsed_s,
                    };
                    return updated;
                  });
                } else if (ev.event === "error") {
                  setToolEvents((prev) => {
                    const idx = [...prev].reverse().findIndex((t) => t.status === "running");
                    if (idx === -1) return prev;
                    const realIdx = prev.length - 1 - idx;
                    const updated = [...prev];
                    updated[realIdx] = { ...updated[realIdx]!, status: "error", error: ev.error };
                    return updated;
                  });
                }
              } catch {
                // ignore malformed event JSON
              }
            }
          } catch {
            // ignore malformed messages
          }
        };

        ws.onclose = () => {
          setConnected(false);
          wsRef.current = null;
          reconnectTimer = setTimeout(connect, 3000);
        };

        ws.onerror = () => {
          setConnected(false);
        };
      } catch {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 3000);
      }
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (ws) {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ op: "unadvertise", topic: inputTopic }));
        }
        ws.onclose = null;
        ws.close();
      }
      wsRef.current = null;
      setConnected(false);
    };
  }, [rosbridgeUrl, inputTopic, outputTopic]);

  // ── Task server status polling ─────────────────────────────────────────────

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${taskServerUrlRef.current}/status`);
        if (res.ok) {
          const data = (await res.json()) as { status: string; embodiment?: string };
          setRobotStatus(data.status === "busy" ? "busy" : "available");
          if (data.embodiment) {
            setEmbodiment(data.embodiment as Embodiment);
          }
        }
      } catch {
        setRobotStatus("unknown");
      }
    };

    poll();
    const id = setInterval(poll, TASK_POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [taskServerUrl]);

  // ── Task result polling ────────────────────────────────────────────────────

  useEffect(() => {
    if (currentTaskId == null) return;

    const poll = async () => {
      try {
        const res = await fetch(`${taskServerUrlRef.current}/tasks/${currentTaskId}/status`);
        if (!res.ok) return;
        const data = (await res.json()) as { status: string; content: string };
        if (data.status === "completed" || data.status === "aborted") {
          setCurrentTaskId(null);
        }
      } catch {
        // ignore — status polling will show unavailability
      }
    };

    const id = setInterval(poll, TASK_POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [currentTaskId, taskServerUrl]);

  // ── Auto-scroll ────────────────────────────────────────────────────────────

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (reasoningRef.current) {
      reasoningRef.current.scrollTop = reasoningRef.current.scrollHeight;
    }
  }, [reasoning]);

  // ── Send / Execute task ────────────────────────────────────────────────────

  const sendMessage = useCallback(async () => {
    const trimmed = inputValue.trim();
    if (trimmed.length === 0) return;

    const sessionHash = generateSessionHash();
    setInputValue("");
    setMessages((prev) => [...prev, { role: "user", content: trimmed, timestamp: Date.now(), sessionHash }]);
    setReasoning("");
    setToolEvents([]);
    setReasoningOpen(true);

    // Try task server first
    try {
      const res = await fetch(`${taskServerUrlRef.current}/execute_task`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: trimmed }),
      });
      if (res.ok) {
        const data = (await res.json()) as { success: boolean; task_id: number | null; content: string };
        if (data.success && data.task_id != null) {
          setCurrentTaskId(data.task_id);
        } else {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: data.content, timestamp: Date.now() },
          ]);
        }
        inputRef.current?.focus();
        return;
      }
    } catch {
      // task server unreachable — fall through
    }

    // Fallback: publish via Foxglove data source
    if (context.publish) {
      try {
        context.publish(COMMAND_TOPIC, { data: trimmed });
      } catch {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "[error] Failed to publish message. Is the data source connected?",
            timestamp: Date.now(),
          },
        ]);
      }
    } else if (wsRef.current?.readyState === WebSocket.OPEN) {
      const commId = `foxglove-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
      wsRef.current.send(
        JSON.stringify({
          op: "publish",
          topic: inputTopic,
          msg: buildHRIMessage(trimmed, commId),
        }),
      );
    } else {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "[offline] Publishing not available. Connect Foxglove to a Rosbridge data source first.",
          timestamp: Date.now(),
        },
      ]);
    }

    inputRef.current?.focus();
  }, [inputValue, context, inputTopic]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void sendMessage();
      }
    },
    [sendMessage],
  );

  // ── Render ─────────────────────────────────────────────────────────────────

  const t = THEMES[colorScheme];

  const robotStatusLabel =
    robotStatus === "available" ? "available" : robotStatus === "busy" ? "busy" : "—";

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <span style={agentStatusDotStyle(agentStatus, connected)} />
        Eastworld-Robotics
        <span style={{ fontSize: "11px", color: THEMES[colorScheme].textMuted, marginLeft: "8px" }}>
          {connected ? "connected" : "no data source"}
        </span>
        {connected && (
          <span
            style={{
              fontSize: "11px",
              color: robotStatus === "available" ? "#4caf50" : robotStatus === "busy" ? "#ff9800" : t.textFaint,
            }}
          >
            {robotStatusLabel}
          </span>
        )}
        <span style={{ fontSize: "10px", color: t.textFaint, marginLeft: "auto", fontWeight: 400 }}>
          {MODEL_OPTIONS.find((o) => o.value === model)?.label ?? model}
        </span>
      </div>

      {/* Reasoning section */}
      <div style={styles.section}>
        <div style={styles.sectionHeader} onClick={() => setReasoningOpen((v) => !v)}>
          <span>{reasoningOpen ? "▾" : "▸"}</span>
          <span>Reasoning</span>
          {agentStatus === "thinking" && (
            <span style={{ color: "#4a9eff", marginLeft: "4px" }}>●</span>
          )}
        </div>
        {reasoningOpen && reasoning && (
          <div ref={reasoningRef} style={{ ...styles.sectionBody, maxHeight: "200px", overflowY: "auto" }}>{reasoning}</div>
        )}
      </div>

      {/* Skill Activity section */}
      <div style={styles.section}>
        <div style={styles.sectionHeader} onClick={() => setToolsOpen((v) => !v)}>
          <span>{toolsOpen ? "▾" : "▸"}</span>
          <span>Skill Activity ({toolEvents.length})</span>
          {agentStatus === "executing" && (
            <span style={{ color: "#ff9800", marginLeft: "4px" }}>●</span>
          )}
        </div>
        {toolsOpen && toolEvents.length > 0 && (
          <div style={styles.sectionBody}>
            {toolEvents.map((ev) => (
              <div key={ev.id} style={{ marginBottom: "2px" }}>
                <span
                  style={{
                    color: ev.status === "done" ? "#4caf50" : ev.status === "error" ? "#f44336" : "#ff9800",
                    marginRight: "4px",
                  }}
                >
                  {ev.status === "done" ? "✓" : ev.status === "error" ? "✗" : "…"}
                </span>
                <span style={{ color: t.text }}>{ev.tool}</span>
                {ev.elapsed_s != null && (
                  <span style={{ color: t.textFaint, marginLeft: "4px" }}>({ev.elapsed_s}s)</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Messages */}
      <div style={{ ...styles.messagesContainer, position: "relative" }}>
        {/* Banner watermark — always visible, fades when messages are present */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            pointerEvents: "none",
            transition: "opacity 0.4s ease",
            opacity: messages.length === 0 ? 1 : 0.06,
            padding: "24px",
          }}
        >
          <img
            src={bannerDataUrl}
            alt=""
            style={{ width: "100%", maxWidth: "380px", height: "auto", objectFit: "contain" }}
          />
          {messages.length === 0 && !connected && (
            <div style={{ fontSize: "11px", marginTop: "4px", color: THEMES[colorScheme].textFaint, textAlign: "center" }}>
              Connect Foxglove to a <strong>Rosbridge</strong> data source to get started.
            </div>
          )}
        </div>
        {messages.length > 0 && (
          <>
            {messages.map((msg) => (
              <div key={msg.timestamp} style={{ display: "flex", flexDirection: "column" }}>
                {msg.role === "user" ? (
                  <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px", alignSelf: "flex-end" }}>
                    <span style={{
                      fontSize: "10px",
                      fontWeight: 700,
                      color: t.accentText,
                      backgroundColor: t.accent,
                      padding: "1px 6px",
                      borderRadius: "4px",
                      letterSpacing: "0.5px",
                      textTransform: "uppercase" as const,
                    }}>
                      ACP
                    </span>
                    <span style={{ fontSize: "10px", color: t.textMuted, fontFamily: "monospace" }}>
                      #{msg.sessionHash ?? msg.communicationId ?? "—"}
                    </span>
                  </div>
                ) : (
                  <div style={styles.roleLabel}>
                    {msg.role === "thought" ? "thought" : embodiment}
                  </div>
                )}
                <div
                  style={
                    msg.role === "user"
                      ? styles.userBubble
                      : msg.role === "thought"
                        ? styles.thoughtBubble
                        : styles.assistantBubble
                  }
                >
                  {msg.content}
                </div>
              </div>
            ))}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div style={styles.inputContainer}>
        <input
          ref={inputRef}
          type="text"
          placeholder="Enter a task for the robot..."
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
          }}
          onKeyDown={handleKeyDown}
          style={styles.input}
        />
        <button
          onClick={() => void sendMessage()}
          disabled={inputValue.trim().length === 0 || robotStatus === "busy"}
          style={{
            ...styles.sendButton,
            opacity: inputValue.trim().length === 0 || robotStatus === "busy" ? 0.5 : 1,
          }}
        >
          Execute
        </button>
      </div>
    </div>
  );
}

export function initAgentPanel(context: PanelExtensionContext): () => void {
  const root = createRoot(context.panelElement);
  root.render(<AgentPanel context={context} />);

  return () => {
    root.unmount();
  };
}
