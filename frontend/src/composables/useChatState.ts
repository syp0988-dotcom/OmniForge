import {
  messages, sessions, currentSessionId, thinking, activeSection, debugData,
  sourceMode, themeMode, languageMode, densityMode, motionMode, switchingSession,
  streamingPhase, streamingCategory, tasks, documents, searchQuery, searchResults,
  uploading, uploadStatus, agents, tools, toolCapabilities, toolExecutor,
  outputFiles, fileProposalStatuses, workspacePath, showFolderReminder, dirEntries,
  browseCurrentPath, workspaceError, collectFilesFromDrop,
} from './chatState'
import { useChatActions } from './useChatActions'
import { useKnowledgeActions } from './useKnowledgeActions'
import { useWorkspaceActions } from './useWorkspaceActions'

/* ------------------------------------------------------------------ */
/*  Aggregated composable: singleton state + domain action groups      */
/* ------------------------------------------------------------------ */

export function useChatState() {
  return {
    // state
    messages, tasks, sessions, currentSessionId, thinking,
    sourceMode, themeMode, languageMode, densityMode, motionMode, switchingSession,
    streamingPhase, streamingCategory, activeSection, debugData,
    documents, searchQuery, searchResults, uploading, uploadStatus,
    agents, tools, toolCapabilities, toolExecutor,
    outputFiles, fileProposalStatuses,
    // workspace state
    workspacePath, showFolderReminder, dirEntries, browseCurrentPath, workspaceError,
    // actions
    ...useChatActions(),
    ...useKnowledgeActions(),
    ...useWorkspaceActions(),
    // file helpers
    collectFilesFromDrop,
  }
}

export type ChatState = ReturnType<typeof useChatState>
