// ==================================
// frontend/components/video-player/hooks/useTranslation.ts
// ==================================
import { useState, useRef } from 'react'
import { toast } from 'sonner'
import { API_BASE_URL, LANGUAGE_MAP } from '../utils/format'
import type { TranslationState, TranslationControls } from '../types'

type TimeoutHandle = ReturnType<typeof setTimeout>

export function useTranslation(onTaskIdChange: (taskId: string | null) => void) {
  // ===================== (更新状态初始化) =====================
  const [state, setState] = useState<TranslationState>({
    isTranslating: false,
    isProcessing: false,
    selectedLanguage: '中文',
    taskId: null,
    selectedFile: null,
    isCompleted: false,
    subtitleWanted: false,
    hlsReady: false,
    // 新增状态
    isUploaded: false,
    isPreprocessing: false,
    isPreprocessed: false
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

  // ================ 切换字幕Wanted状态 ==================
  const toggleSubtitleWanted = () => {
    setState(prev => ({ ...prev, subtitleWanted: !prev.subtitleWanted }))
  }

  // ================ 开始轮询任务状态 ==================
  const startPolling = (taskId: string, stopStatuses: string[] = ['success', 'error']) => {
    // 清除旧的轮询
    stopPolling()

    // 首先立即执行一次状态检查
    fetchTaskStatus(taskId).then(statusData => {
      if (statusData) {
        handleTaskStatus(statusData, taskId, stopStatuses);
      }
    });

    // 开始新的轮询
    pollIntervalRef.current = setInterval(async () => {
      try {
        const statusData = await fetchTaskStatus(taskId);
        if (statusData) {
          handleTaskStatus(statusData, taskId, stopStatuses);
        }
      } catch (error) {
        console.error('轮询任务状态错误:', error)
      }
    }, 5000)
  }

  // 抽取任务状态获取函数
  const fetchTaskStatus = async (taskId: string) => {
    try {
      const statusResponse = await fetch(`${API_BASE_URL}/task/${taskId}`)
      if (!statusResponse.ok) return null;
      return await statusResponse.json();
    } catch (error) {
      console.error('获取任务状态失败:', error)
      return null;
    }
  }

  // 处理任务状态变化
  const handleTaskStatus = (statusData: any, taskId: string, stopStatuses: string[]) => {
    // 检查HLS就绪状态，只在首次检测到时初始化播放器
    if (statusData.hls_ready && !hlsInitializedRef.current) {
      console.log('HLS播放列表已就绪，初始化播放器')
      hlsInitializedRef.current = true
      setState(prev => ({ ...prev, hlsReady: true }))
      
      // 初始化HLS播放器
      onTaskIdChange(taskId)
    }

    console.log('当前任务状态:', statusData.status);

    // 根据状态更新本地状态
    if (statusData.status === 'uploaded') {
      setState(prev => ({
        ...prev,
        isUploaded: true,
        isProcessing: false
      }))
    } else if (statusData.status === 'preprocessing') {
      setState(prev => ({
        ...prev,
        isPreprocessing: true,
        isPreprocessed: false,
        isProcessing: true
      }))
    } else if (statusData.status === 'preprocessed') {
      setState(prev => ({
        ...prev,
        isPreprocessing: false,
        isPreprocessed: true,
        isProcessing: false
      }))
      if (stopStatuses.includes('preprocessed')) {
        stopPolling()
      }
    } else if (statusData.status === 'translating' || statusData.status === 'mixing') {
      setState(prev => ({
        ...prev,
        isTranslating: true,
        isProcessing: true
      }))
    } else if (statusData.status === 'success') {
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
        hlsReady: false,
        isPreprocessing: false
      }))
      hlsInitializedRef.current = false
      onTaskIdChange(null)
      stopPolling()
    }
    
    // 检查是否需要停止轮询
    if (stopStatuses.includes(statusData.status)) {
      stopPolling()
    }
  }

  const controls: TranslationControls = {
    // ================ 上传视频 ==================
    uploadVideo: async () => {
      if (!state.selectedFile) {
        toast.error('请先上传视频')
        return
      }

      setState(prev => ({ 
        ...prev, 
        isProcessing: true,
        isUploaded: false,
        isPreprocessed: false,
        isPreprocessing: false,
        isTranslating: false,
        isCompleted: false, 
        hlsReady: false,
        taskId: null
      }))
      
      // 重置HLS初始化状态
      hlsInitializedRef.current = false

      const formData = new FormData()
      formData.append('video', state.selectedFile)
      // 确保target_language不为null
      formData.append('target_language', LANGUAGE_MAP[state.selectedLanguage] || 'zh')
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
        
        // 检查data是否包含status字段，并确保与后端返回的是一致的
        const status = data.status || 'uploaded';
        
        setState(prev => ({
          ...prev,
          taskId: data.task_id,
          isUploaded: status === 'uploaded',
          isProcessing: false
        }))
        
        toast.success('视频上传成功')
        
        // 可以立即启动一次状态检查，确保状态是最新的
        if (data.task_id) {
          try {
            const statusResponse = await fetch(`${API_BASE_URL}/task/${data.task_id}`)
            if (statusResponse.ok) {
              const statusData = await statusResponse.json()
              // 根据任务状态更新UI状态
              if (statusData.status === 'uploaded') {
                setState(prev => ({
                  ...prev,
                  isUploaded: true,
                  isProcessing: false
                }))
              }
            }
          } catch (error) {
            console.error('获取上传任务状态失败:', error)
          }
        }
      } catch (error) {
        console.error('上传视频错误:', error)
        toast.error('上传视频失败，请重试')
        setState(prev => ({ 
          ...prev, 
          isProcessing: false,
          isUploaded: false
        }))
      }
    },

    // ================ 预处理视频 ==================
    preprocessVideo: async () => {
      if (!state.taskId || !state.isUploaded) {
        toast.error('请先上传视频')
        return
      }

      setState(prev => ({ 
        ...prev,
        isPreprocessing: true,
        isPreprocessed: false,
        isProcessing: true
      }))

      const formData = new FormData()
      // 确保传递target_language
      formData.append('target_language', LANGUAGE_MAP[state.selectedLanguage] || 'zh')
      // 确保传递generate_subtitle
      formData.append('generate_subtitle', state.subtitleWanted ? 'true' : 'false')

      try {
        const response = await fetch(`${API_BASE_URL}/preprocess/${state.taskId}`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || '预处理失败')
        }

        const data = await response.json()
        
        // 更新状态为预处理中
        setState(prev => ({
          ...prev,
          isPreprocessing: true
        }))
        
        toast.success('开始预处理视频')
        
        // 开始轮询，当状态变为 preprocessed 时停止
        startPolling(state.taskId, ['preprocessed', 'error'])
      } catch (error) {
        console.error('预处理视频错误:', error)
        toast.error('预处理视频失败，请重试')
        setState(prev => ({ 
          ...prev, 
          isPreprocessing: false,
          isProcessing: false
        }))
      }
    },

    // ================ 翻译视频 ==================
    startTranslation: async () => {
      if (!state.taskId || !state.isPreprocessed) {
        toast.error('请先完成视频预处理')
        return
      }

      setState(prev => ({ 
        ...prev,
        isTranslating: true,
        isProcessing: true
      }))

      try {
        const response = await fetch(`${API_BASE_URL}/translate/${state.taskId}`, {
          method: 'POST'
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || '翻译失败')
        }

        const data = await response.json()
        
        // 更新状态
        setState(prev => ({
          ...prev,
          isTranslating: true
        }))
        
        toast.success('开始翻译视频')
        
        // 开始轮询，直到任务成功或失败
        startPolling(state.taskId, ['success', 'error'])
      } catch (error) {
        console.error('翻译错误:', error)
        toast.error('翻译失败，请重试')
        setState(prev => ({ 
          ...prev, 
          isTranslating: false,
          isProcessing: false 
        }))
      }
    },

    stopTranslation: () => {
      setState(prev => ({
        ...prev,
        isTranslating: false,
        isProcessing: false,
        isPreprocessing: false,
        isPreprocessed: false,
        isUploaded: false,
        taskId: null,
        isCompleted: false,
        hlsReady: false
      }))
      hlsInitializedRef.current = false
      onTaskIdChange(null)
      stopPolling()
      toast.success('已停止处理')
    },

    setLanguage: (language: string) => {
      setState(prev => ({ ...prev, selectedLanguage: language }))
    },

    toggleSubtitleWanted,
  }

  return {
    state,
    setState,
    controls
  }
}