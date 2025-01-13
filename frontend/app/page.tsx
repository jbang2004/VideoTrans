"use client";

import { Upload, Link2, Globe } from 'lucide-react'
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useState, useRef } from 'react'
import { toast } from "sonner"
import UploadArea from '@/components/UploadArea'
import VideoPlayer from '@/components/VideoPlayer'
import { Oval } from 'react-loader-spinner' // 引入加载动画

const API_BASE_URL = 'http://localhost:8000'; // 后端服务地址

export default function VideoTranslationPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [targetLanguage, setTargetLanguage] = useState("en");
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null); // 添加 ref

  // 上传视频并开始处理
  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      setIsUploading(true)
      setUploadProgress(0)
      const formData = new FormData()
      formData.append('video', selectedFile)
      formData.append('target_language', targetLanguage)

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || '上传失败')
      }

      const data = await response.json()
      
      if (data.task_id) {
        setTaskId(data.task_id)
        toast.success('视频上传成功，开始处理')
        startPolling(data.task_id)
      }
    } catch (error) {
      console.error('Upload error:', error)
      toast.error(error instanceof Error ? error.message : '上传失败')
    } finally {
      setIsUploading(false)
    }
  }

  // 轮询任务状态
  const startPolling = (taskId: string) => {
    const pollInterval = 5000
    const maxRetries = 60
    let retryCount = 0

    const checkStatus = async () => {
      try {
        if (retryCount >= maxRetries) {
          toast.error('处理超时')
          return
        }

        const response = await fetch(`${API_BASE_URL}/task/${taskId}`)
        const data = await response.json()

        if (data.status === 'success') {
          toast.success('视频处理完成')
          setUploadProgress(100)
          return
        } else if (data.status === 'error') {
          toast.error(`处理失败: ${data.message}`)
          return
        }

        // 更新进度
        if (data.progress) {
          setUploadProgress(data.progress)
        }

        retryCount++
        setTimeout(checkStatus, pollInterval)
      } catch (error) {
        console.error('Poll error:', error)
        setTimeout(checkStatus, pollInterval)
      }
    }

    checkStatus()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 flex flex-col">
      <header className="py-6 bg-gradient-to-r from-purple-600 to-blue-600">
        <h1 className="text-4xl font-bold text-center text-white tracking-tight">
          视频闪译
        </h1>
      </header>

      <div className="flex flex-1">
        {/* Sidebar */}
        <div className="w-full lg:w-80 bg-purple-50/80 p-6 flex flex-col">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6">上传视频</h2>
          <Button
            variant="ghost"
            className="w-full justify-start gap-3 text-gray-700 hover:bg-purple-100/50 active:bg-purple-200/50 active:scale-95 transition-all duration-200"
            onClick={() => {
              // 触发文件选择
              fileInputRef.current?.click()
            }}
          >
            <Upload className="w-4 h-4" />
            上传视频
          </Button>
          <Button 
            variant="ghost" 
            className="w-full justify-start gap-3 text-gray-700 hover:bg-purple-100/50 active:bg-purple-200/50 active:scale-95 transition-all duration-200"
          >
            <Link2 className="w-4 h-4" />
            使用视频链接
          </Button>
        </div>

        {/* Main content */}
        <div className="flex-1 flex justify-center items-center p-8">
          <div className="w-full max-w-4xl bg-white/70 backdrop-blur-sm rounded-2xl shadow-xl border border-purple-100 p-8">
            <div className="flex justify-between items-center mb-6">
              <p className="text-gray-600">
                {selectedFile ? `已选择: ${selectedFile.name}` : '上传您想要翻译的视频'}
              </p>
              <div className="flex items-center gap-4">
                <Button 
                  variant="outline" 
                  className="gap-2 hover:bg-purple-50/50 active:bg-purple-100/50 active:scale-95 transition-all duration-200"
                  onClick={() => {
                    // 触发文件选择
                    fileInputRef.current?.click()
                  }}
                >
                  <svg
                    viewBox="0 0 24 24"
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <path d="M12 2v20M2 12h20" />
                  </svg>
                  上传视频
                </Button>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="hover:bg-gray-100/50 active:bg-gray-200/50 active:scale-95 transition-all duration-200"
                >
                  提示
                </Button>
              </div>
            </div>
            
            {/* 上传区域或视频播放器 */}
            {!taskId ? (
              <UploadArea onFileSelect={(file) => setSelectedFile(file)} />
            ) : (
              <VideoPlayer taskId={taskId} apiBaseUrl={API_BASE_URL} />
            )}

            {/* 操作按钮 */}
            <div className="flex items-center justify-end gap-4 mt-8">
              <Select 
                defaultValue={targetLanguage}
                onValueChange={setTargetLanguage}
              >
                <SelectTrigger className="w-[180px] bg-white/50 hover:bg-purple-50/30 transition-colors duration-200">
                  <Globe className="w-4 h-4 mr-2" />
                  <SelectValue placeholder="选择语言" />
                </SelectTrigger>
                <SelectContent className="bg-white/95 backdrop-blur-sm border-purple-100">
                  <SelectItem value="zh" className="hover:bg-purple-50/80 focus:bg-purple-50/80 cursor-pointer transition-colors duration-150">中文</SelectItem>
                  <SelectItem value="en" className="hover:bg-purple-50/80 focus:bg-purple-50/80 cursor-pointer transition-colors duration-150">英语</SelectItem>
                  <SelectItem value="ja" className="hover:bg-purple-50/80 focus:bg-purple-50/80 cursor-pointer transition-colors duration-150">日语</SelectItem>
                  <SelectItem value="ko" className="hover:bg-purple-50/80 focus:bg-purple-50/80 cursor-pointer transition-colors duration-150">韩语</SelectItem>
                </SelectContent>
              </Select>

              <Button 
                className="bg-purple-600 hover:bg-purple-700 active:bg-purple-800 text-white active:scale-95 transition-all duration-200 flex items-center justify-center"
                onClick={handleUpload}
                disabled={!selectedFile || isUploading}
              >
                {isUploading ? (
                  <>
                    <Oval
                      height={20}
                      width={20}
                      color="#ffffff"
                      wrapperStyle={{ marginRight: '8px' }}
                      visible={true}
                      ariaLabel='oval-loading'
                      secondaryColor="#ffffff"
                      strokeWidth={2}
                      strokeWidthSecondary={2}
                    />
                    处理中...
                  </>
                ) : '开始翻译'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}