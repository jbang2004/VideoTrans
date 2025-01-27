// ==================================
// frontend/components/video-player/hooks/useTranslation.ts
// ==================================
import { useState, useRef } from 'react'
import { toast } from 'sonner'
import { API_BASE_URL, LANGUAGE_MAP } from '../utils/format'
import type { TranslationState, TranslationControls } from '../types'

type TimeoutHandle = ReturnType<typeof setTimeout>

export function useTranslation(onTaskIdChange: (taskId: string | null) => void) {
  // ===================== (在这里初始化 subtitleWanted) =====================
  const [state, setState] = useState<TranslationState>({
    isTranslating: false,
    isProcessing: false,
    selectedLanguage: '中文',
    taskId: null,
    selectedFile: null,
    isCompleted: false,
    subtitleWanted: false, // 新增
  })

  const pollIntervalRef = useRef<TimeoutHandle>()

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = undefined
    }
  }

  // ================ (新增) 切换字幕Wanted状态 ==================
  const toggleSubtitleWanted = () => {
    setState(prev => ({ ...prev, subtitleWanted: !prev.subtitleWanted }))
  }

  const controls: TranslationControls = {
    startTranslation: async () => {
      if (!state.selectedFile) {
        toast.error('请先上传视频')
        return
      }

      setState(prev => ({ ...prev, isProcessing: true, isCompleted: false }))

      const formData = new FormData()
      formData.append('video', state.selectedFile)
      formData.append('target_language', LANGUAGE_MAP[state.selectedLanguage] || 'zh')
      // =============== (关键) 传递 generate_subtitle = subtitleWanted ================
      formData.append('generate_subtitle', state.subtitleWanted ? 'true' : 'false')

      try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || '上传失败')
        }

        const data = await response.json()
        setState(prev => ({
          ...prev,
          taskId: data.task_id,
          isTranslating: true
        }))
        onTaskIdChange(data.task_id)
        toast.success('开始翻译处理')

        // 开始轮询任务状态
        pollIntervalRef.current = setInterval(async () => {
          try {
            const statusResponse = await fetch(`${API_BASE_URL}/task/${data.task_id}`)
            if (!statusResponse.ok) return

            const statusData = await statusResponse.json()
            if (statusData.status === 'success') {
              toast.success('视频翻译完成')
              setState(prev => ({
                ...prev,
                isProcessing: false,
                isCompleted: true,
              }))
              stopPolling()
            } else if (statusData.status === 'error') {
              toast.error(statusData.message || '处理失败')
              setState(prev => ({
                ...prev,
                isTranslating: false,
                isProcessing: false,
                isCompleted: false
              }))
              stopPolling()
            }
          } catch (error) {
            console.error('轮询任务状态错误:', error)
          }
        }, 5000)
      } catch (error) {
        console.error('开始翻译错误:', error)
        toast.error('开始翻译失败，请重试')
        setState(prev => ({ ...prev, isProcessing: false }))
      }
    },

    stopTranslation: () => {
      setState(prev => ({
        ...prev,
        isTranslating: false,
        isProcessing: false,
        taskId: null,
        isCompleted: false
      }))
      onTaskIdChange(null)
      stopPolling()
      toast.success('已停止翻译')
    },

    setLanguage: (language: string) => {
      setState(prev => ({ ...prev, selectedLanguage: language }))
    },

    // ============== (新增) ==================
    toggleSubtitleWanted,
  }

  return {
    state,
    setState,
    controls
  }
}
