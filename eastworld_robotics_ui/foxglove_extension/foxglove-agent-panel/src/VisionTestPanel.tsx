import {
  Immutable,
  PanelExtensionContext,
  RenderState,
  SettingsTreeAction,
} from "@foxglove/extension";
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

// ── Types ───────────────────────────────────────────────────────────────────────

type ColorScheme = "dark" | "light";
type NodeStatus = "disconnected" | "waiting_for_camera" | "ready" | "processing" | "speaking" | "error";

interface ThemeColors {
  bg: string;
  bgSurface: string;
  bgElevated: string;
  text: string;
  textMuted: string;
  textFaint: string;
  border: string;
  accent: string;
  accentText: string;
  successBg: string;
  successText: string;
  errorBg: string;
  errorText: string;
}

// ── Constants ───────────────────────────────────────────────────────────────────

const DEFAULT_ROSBRIDGE_URL = "ws://localhost:9090";
const DEFAULT_PROMPT_TOPIC = "/vlm_prompt";
const DEFAULT_RESPONSE_TOPIC = "/vlm_response";
const DEFAULT_STATUS_TOPIC = "/vlm_status";
const DEFAULT_REASONING_TOPIC = "/vlm_reasoning";

const DEFAULT_PROMPT =
  "Describe what you see in this image. " +
  "Identify key objects, their spatial relationships, and anything notable.";

// ── Theme ───────────────────────────────────────────────────────────────────────

const THEMES: Record<ColorScheme, ThemeColors> = {
  dark: {
    bg: "#1a1a1a",
    bgSurface: "#242424",
    bgElevated: "#2d2d2d",
    text: "#e0e0e0",
    textMuted: "#999",
    textFaint: "#666",
    border: "#333",
    accent: "#4a9eff",
    accentText: "#fff",
    successBg: "#1b3a2a",
    successText: "#4caf50",
    errorBg: "#3a1b1b",
    errorText: "#f44336",
  },
  light: {
    bg: "#fafafa",
    bgSurface: "#ffffff",
    bgElevated: "#f0f0f0",
    text: "#1a1a1a",
    textMuted: "#666",
    textFaint: "#999",
    border: "#e0e0e0",
    accent: "#1976d2",
    accentText: "#fff",
    successBg: "#e8f5e9",
    successText: "#2e7d32",
    errorBg: "#ffebee",
    errorText: "#c62828",
  },
};

const STATUS_LABELS: Record<NodeStatus, string> = {
  disconnected: "Disconnected",
  waiting_for_camera: "Waiting for camera",
  ready: "Ready",
  processing: "Processing...",
  speaking: "Speaking...",
  error: "Error",
};

function statusColor(status: NodeStatus, t: ThemeColors): string {
  switch (status) {
    case "ready":
      return t.successText;
    case "processing":
      return t.accent;
    case "speaking":
      return "#ab47bc";
    case "error":
      return t.errorText;
    case "waiting_for_camera":
      return "#ff9800";
    default:
      return t.textFaint;
  }
}

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
      padding: "12px 16px",
      borderBottom: `1px solid ${t.border}`,
      display: "flex",
      alignItems: "center",
      gap: "10px",
      flexShrink: 0,
    } satisfies CSSProperties,
    title: {
      fontWeight: 700,
      fontSize: "14px",
      letterSpacing: "-0.01em",
    } satisfies CSSProperties,
    promptSection: {
      padding: "12px 16px",
      borderBottom: `1px solid ${t.border}`,
      flexShrink: 0,
    } satisfies CSSProperties,
    promptLabel: {
      fontSize: "11px",
      fontWeight: 600,
      color: t.textMuted,
      textTransform: "uppercase" as const,
      letterSpacing: "0.5px",
      marginBottom: "6px",
    } satisfies CSSProperties,
    promptTextarea: {
      width: "100%",
      minHeight: "60px",
      maxHeight: "120px",
      padding: "8px 10px",
      backgroundColor: t.bgSurface,
      color: t.text,
      border: `1px solid ${t.border}`,
      borderRadius: "6px",
      outline: "none",
      fontSize: "13px",
      fontFamily: "inherit",
      resize: "vertical" as const,
      lineHeight: "1.5",
      boxSizing: "border-box" as const,
    } satisfies CSSProperties,
    buttonRow: {
      display: "flex",
      gap: "8px",
      marginTop: "8px",
    } satisfies CSSProperties,
    runButton: {
      padding: "7px 20px",
      backgroundColor: t.accent,
      color: t.accentText,
      border: "none",
      borderRadius: "6px",
      cursor: "pointer",
      fontWeight: 600,
      fontSize: "13px",
      transition: "opacity 0.15s",
    } satisfies CSSProperties,
    responseSection: {
      flex: 1,
      overflow: "hidden",
      display: "flex",
      flexDirection: "column" as const,
    } satisfies CSSProperties,
    responseHeader: {
      padding: "10px 16px 0",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      flexShrink: 0,
    } satisfies CSSProperties,
    responseLabel: {
      fontSize: "11px",
      fontWeight: 600,
      color: t.textMuted,
      textTransform: "uppercase" as const,
      letterSpacing: "0.5px",
    } satisfies CSSProperties,
    responseBody: {
      flex: 1,
      overflowY: "auto" as const,
      padding: "8px 16px 16px",
    } satisfies CSSProperties,
    responseText: {
      backgroundColor: t.bgSurface,
      border: `1px solid ${t.border}`,
      borderRadius: "6px",
      padding: "12px",
      lineHeight: "1.6",
      whiteSpace: "pre-wrap" as const,
      wordBreak: "break-word" as const,
      fontSize: "13px",
    } satisfies CSSProperties,
    emptyState: {
      display: "flex",
      flexDirection: "column" as const,
      alignItems: "center",
      justifyContent: "center",
      flex: 1,
      color: t.textFaint,
      gap: "6px",
      padding: "24px",
      textAlign: "center" as const,
    } satisfies CSSProperties,
    timestamp: {
      fontSize: "11px",
      color: t.textFaint,
    } satisfies CSSProperties,
  };
}

// ── Component ───────────────────────────────────────────────────────────────────

function VisionTestPanel({ context }: { context: PanelExtensionContext }): ReactElement {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [response, setResponse] = useState<string | null>(null);
  const [responseTime, setResponseTime] = useState<string | null>(null);
  const [reasoning, setReasoning] = useState<string | null>(null);
  const [reasoningOpen, setReasoningOpen] = useState(false);
  const [nodeStatus, setNodeStatus] = useState<NodeStatus>("disconnected");
  const [connected, setConnected] = useState(false);
  const [rosbridgeUrl, setRosbridgeUrl] = useState(DEFAULT_ROSBRIDGE_URL);
  const [promptTopic, setPromptTopic] = useState(DEFAULT_PROMPT_TOPIC);
  const [responseTopic, setResponseTopic] = useState(DEFAULT_RESPONSE_TOPIC);
  const [statusTopic, setStatusTopic] = useState(DEFAULT_STATUS_TOPIC);
  const [renderDone, setRenderDone] = useState<(() => void) | undefined>();
  const [colorScheme, setColorScheme] = useState<ColorScheme>("dark");

  const wsRef = useRef<WebSocket | null>(null);
  const responseTopicRef = useRef(responseTopic);
  responseTopicRef.current = responseTopic;
  const statusTopicRef = useRef(statusTopic);
  statusTopicRef.current = statusTopic;
  const reasoningTopicRef = useRef(DEFAULT_REASONING_TOPIC);
  reasoningTopicRef.current = DEFAULT_REASONING_TOPIC;

  const theme = THEMES[colorScheme];
  const styles = useMemo(() => buildStyles(theme), [colorScheme]);

  // ── Foxglove render handler ───────────────────────────────────────────────

  useLayoutEffect(() => {
    context.onRender = (renderState: Immutable<RenderState>, done: () => void) => {
      if (renderState.colorScheme) {
        setColorScheme(renderState.colorScheme);
      }
      setRenderDone(() => done);
    };
    context.watch("colorScheme");
  }, [context]);

  useEffect(() => {
    renderDone?.();
  }, [renderDone]);

  // ── Settings ──────────────────────────────────────────────────────────────

  const settingsActionHandler = useCallback((action: SettingsTreeAction) => {
    if (action.action === "update") {
      const { path, value } = action.payload;
      if (typeof value !== "string" || value.length === 0) return;
      switch (path[1]) {
        case "rosbridgeUrl":
          setRosbridgeUrl(value);
          break;
        case "promptTopic":
          setPromptTopic(value);
          break;
        case "responseTopic":
          setResponseTopic(value);
          break;
        case "statusTopic":
          setStatusTopic(value);
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
            rosbridgeUrl: { label: "Rosbridge URL", input: "string", value: rosbridgeUrl },
          },
        },
        topics: {
          label: "Topics",
          fields: {
            promptTopic: { label: "Prompt topic (publish)", input: "string", value: promptTopic },
            responseTopic: { label: "Response topic (subscribe)", input: "string", value: responseTopic },
            statusTopic: { label: "Status topic (subscribe)", input: "string", value: statusTopic },
          },
        },
      },
    });
  }, [context, rosbridgeUrl, promptTopic, responseTopic, statusTopic, settingsActionHandler]);

  // ── WebSocket ─────────────────────────────────────────────────────────────

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      try {
        ws = new WebSocket(rosbridgeUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          // Subscribe to response topic
          ws.send(JSON.stringify({
            op: "subscribe", topic: responseTopic, type: "std_msgs/msg/String",
          }));
          // Subscribe to status topic
          ws.send(JSON.stringify({
            op: "subscribe", topic: statusTopic, type: "std_msgs/msg/String",
          }));
          // Subscribe to reasoning topic
          ws.send(JSON.stringify({
            op: "subscribe", topic: DEFAULT_REASONING_TOPIC, type: "std_msgs/msg/String",
          }));
          // Advertise prompt topic
          ws.send(JSON.stringify({
            op: "advertise", topic: promptTopic, type: "std_msgs/msg/String",
          }));
        };

        ws.onmessage = (event: MessageEvent) => {
          try {
            const data = JSON.parse(event.data as string);
            if (data.op !== "publish" || !data.msg?.data) return;

            if (data.topic === responseTopicRef.current) {
              setResponse(data.msg.data);
              setResponseTime(new Date().toLocaleTimeString());
            } else if (data.topic === statusTopicRef.current) {
              const s = data.msg.data as string;
              if (s in STATUS_LABELS) {
                setNodeStatus(s as NodeStatus);
                if (s === "processing") {
                  setReasoning(null);
                  setReasoningOpen(false);
                }
              }
            } else if (data.topic === reasoningTopicRef.current) {
              setReasoning(data.msg.data);
              setReasoningOpen(true);
            }
          } catch { /* ignore */ }
        };

        ws.onclose = () => {
          setConnected(false);
          setNodeStatus("disconnected");
          wsRef.current = null;
          reconnectTimer = setTimeout(connect, 3000);
        };

        ws.onerror = () => setConnected(false);
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
          ws.send(JSON.stringify({ op: "unadvertise", topic: promptTopic }));
        }
        ws.onclose = null;
        ws.close();
      }
      wsRef.current = null;
      setConnected(false);
    };
  }, [rosbridgeUrl, promptTopic, responseTopic, statusTopic]);

  // ── Send prompt ───────────────────────────────────────────────────────────

  const sendPrompt = useCallback(() => {
    const trimmed = prompt.trim();
    if (!trimmed || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(JSON.stringify({
      op: "publish",
      topic: promptTopic,
      msg: { data: trimmed },
    }));
  }, [prompt, promptTopic]);

  const isProcessing = nodeStatus === "processing";
  const canRun = connected && prompt.trim().length > 0 && !isProcessing;

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <span style={{
          display: "inline-block", width: 8, height: 8, borderRadius: "50%",
          backgroundColor: statusColor(connected ? nodeStatus : "disconnected", theme),
        }} />
        <span style={styles.title}>VLM Test</span>
        <span style={{ fontSize: "11px", color: theme.textMuted }}>
          {STATUS_LABELS[connected ? nodeStatus : "disconnected"]}
        </span>
      </div>

      {/* Prompt */}
      <div style={styles.promptSection}>
        <div style={styles.promptLabel}>Prompt</div>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          style={styles.promptTextarea}
          placeholder="Enter your prompt..."
        />
        <div style={styles.buttonRow}>
          <button
            onClick={sendPrompt}
            disabled={!canRun}
            style={{ ...styles.runButton, opacity: canRun ? 1 : 0.4, cursor: canRun ? "pointer" : "default" }}
          >
            {isProcessing ? "Processing..." : "Run"}
          </button>
        </div>
      </div>

      {/* Response */}
      <div style={styles.responseSection}>
        <div style={styles.responseHeader}>
          <span style={styles.responseLabel}>Response</span>
          {responseTime && <span style={styles.timestamp}>{responseTime}</span>}
        </div>

        {response != null ? (
          <div style={styles.responseBody}>
            {reasoning != null && (
              <div style={{ marginBottom: "8px" }}>
                <button
                  onClick={() => setReasoningOpen((o) => !o)}
                  style={{
                    background: "none", border: "none", cursor: "pointer",
                    color: theme.textMuted, fontSize: "11px", fontWeight: 600,
                    textTransform: "uppercase" as const, letterSpacing: "0.5px",
                    padding: "0 0 4px 0", display: "flex", alignItems: "center", gap: "4px",
                  }}
                >
                  {reasoningOpen ? "▾" : "▸"} Reasoning
                </button>
                {reasoningOpen && (
                  <div style={{
                    ...styles.responseText,
                    fontSize: "12px",
                    color: theme.textMuted,
                    backgroundColor: theme.bgElevated,
                    maxHeight: "160px",
                    overflowY: "auto" as const,
                  }}>
                    {reasoning}
                  </div>
                )}
              </div>
            )}
            <div style={styles.responseText}>{response}</div>
          </div>
        ) : (
          <div style={styles.emptyState}>
            <div style={{ fontSize: "20px", opacity: 0.5 }}>&#x1F50D;</div>
            <div>No response yet</div>
            <div style={{ fontSize: "11px" }}>
              Edit the prompt above and click <strong>Run</strong>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function initVisionTestPanel(context: PanelExtensionContext): () => void {
  const root = createRoot(context.panelElement);
  root.render(<VisionTestPanel context={context} />);
  return () => { root.unmount(); };
}
