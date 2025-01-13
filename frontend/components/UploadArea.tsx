// frontend/components/UploadArea.tsx
"use client"

import { useState, useRef } from 'react'

interface UploadAreaProps {
  onFileSelect: (file: File) => void
}

export default function UploadArea({ onFileSelect }: UploadAreaProps) {
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave" || e.type === "drop") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('video/')) {
      onFileSelect(file)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileSelect(file)
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div 
      className={`group p-8 bg-white/50 rounded-xl border-2 border-dashed ${
        dragActive ? 'border-purple-500 bg-purple-50/20' : 'border-gray-200'
      } transition-all duration-300 hover:border-purple-300 hover:shadow-lg cursor-pointer`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept="video/mp4,video/mov,video/webm"
        onChange={handleFileSelect}
      />
      <div className="flex flex-col items-center justify-center min-h-[400px]">
        <div className="p-4 rounded-xl bg-gray-50/50 transition-transform duration-300 group-hover:scale-110">
          <svg
            viewBox="0 0 24 24"
            className="w-16 h-16 text-gray-400"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
            <line x1="7" y1="2" x2="7" y2="22" />
            <line x1="17" y1="2" x2="17" y2="22" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <line x1="2" y1="7" x2="7" y2="7" />
            <line x1="2" y1="17" x2="7" y2="17" />
            <line x1="17" y1="17" x2="22" y2="17" />
            <line x1="17" y1="7" x2="22" y2="7" />
          </svg>
        </div>
        <h3 className="mt-6 text-lg font-medium text-gray-700">
          拖放视频到此处上传
        </h3>
        <p className="mt-2 text-sm text-gray-500">
          文件类型：mp4、mov、webm。文件大小最大 5 GB
        </p>
      </div>
    </div>
  )
}