import { ExtensionContext } from "@foxglove/extension";

import { initAgentPanel } from "./AgentPanel";
import { initVisionTestPanel } from "./VisionTestPanel";

export function activate(extensionContext: ExtensionContext): void {
  extensionContext.registerPanel({ name: "agent-panel", initPanel: initAgentPanel });
  extensionContext.registerPanel({ name: "vision-test", initPanel: initVisionTestPanel });
}
