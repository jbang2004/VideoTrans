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
    hlsReady: false, // 新增：HLS播放列表是否就绪
  })

  const pollIntervalRef = useRef<TimeoutHandle>()
  // 添加一个ref来跟踪HLS是否已经初始化
  const hlsInitializedRef = useRef<boolean>(false)

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

      setState(prev => ({ 
        ...prev, 
        isProcessing: true, 
        isCompleted: false, 
        hlsReady: false // 重置HLS就绪状态
      }))
      // 重置HLS初始化状态
      hlsInitializedRef.current = false

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
        // 不立即初始化HLS播放器，等待hls_ready为true时再初始化
        // onTaskIdChange(data.task_id)
        toast.success('开始翻译处理')

        // 开始轮询任务状态
        pollIntervalRef.current = setInterval(async () => {
          try {
            const statusResponse = await fetch(`${API_BASE_URL}/task/${data.task_id}`)
            if (!statusResponse.ok) return

            const statusData = await statusResponse.json()
            
            // 检查HLS就绪状态，只在首次检测到时初始化播放器
            if (statusData.hls_ready && !hlsInitializedRef.current) {
              console.log('HLS播放列表已就绪，初始化播放器')
              hlsInitializedRef.current = true // 标记为已初始化
              setState(prev => ({ ...prev, hlsReady: true }))
              
              // 简单直接地初始化HLS播放器，不需要复杂的延迟逻辑
              onTaskIdChange(data.task_id)
            }
            
            if (statusData.status === 'success') {
              toast.success('视频翻译完成')
              setState(prev => ({
                ...prev,
                isProcessing: false,
                isCompleted: true,
                hlsReady: true
              }))
              stopPolling()
            } else if (statusData.status === 'error') {
              toast.error(statusData.message || '处理失败')
              setState(prev => ({
                ...prev,
                isTranslating: false,
                isProcessing: false,
                isCompleted: false,
                hlsReady: false
              }))
              hlsInitializedRef.current = false // 重置HLS初始化状态
              onTaskIdChange(null) // 清除任务ID，避免HLS尝试加载
              stopPolling()
            }
          } catch (error) {
            console.error('轮询任务状态错误:', error)
          }
        }, 5000) // 轮询间隔为5秒，避免过于频繁的API请求
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
        isCompleted: false,
        hlsReady: false // 重置HLS就绪状态
      }))
      hlsInitializedRef.current = false // 重置HLS初始化状态
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
