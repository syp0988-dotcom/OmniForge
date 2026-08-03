import { setWorkspace, createServerFolder, browseDirectory } from '@/api/client'
import { workspacePath, showFolderReminder, dirEntries, browseCurrentPath, workspaceError } from './chatState'

export function useWorkspaceActions() {
  const handleSetWorkspace = async (path: string) => {
    try {
      workspaceError.value = null
      const result = await setWorkspace(path)
      workspacePath.value = result.path
      localStorage.setItem('workspace_path', result.path)
      showFolderReminder.value = false
      return result.path
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      workspaceError.value = msg
      throw e
    }
  }
  const handleCreateFolder = async (parentPath: string, folderName: string) => {
    const result = await createServerFolder(parentPath, folderName)
    // Auto-set as workspace
    workspacePath.value = result.path
    localStorage.setItem('workspace_path', result.path)
    showFolderReminder.value = false
    workspaceError.value = null
    return result.path
  }
  const handleBrowse = async (path: string = '.') => {
    try {
      const data = await browseDirectory(path)
      browseCurrentPath.value = data.current_path
      dirEntries.value = data.entries
    } catch (e) {
      console.warn('Failed to browse directory:', e)
      dirEntries.value = []
    }
  }
  const clearWorkspace = () => {
    workspacePath.value = null
    localStorage.removeItem('workspace_path')
    dirEntries.value = []
    browseCurrentPath.value = ''
    workspaceError.value = null
  }

  return { handleSetWorkspace, handleCreateFolder, handleBrowse, clearWorkspace }
}
