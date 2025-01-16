import React, { useRef } from 'react'
import { Button } from '../../ui/button'
import { Upload } from 'lucide-react'
import { toast } from 'sonner'

interface UploadButtonProps {
  onFileSelect: (file: File) => void
  className?: string
}

export function UploadButton({ onFileSelect, className }: UploadButtonProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    onFileSelect(file)
    toast.success('视频已选择，请点击开始翻译')
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileUpload}
      />
      <Button 
        size="icon" 
        variant="ghost" 
        className={className}
        onClick={handleUploadClick}
      >
        <Upload className="h-4 w-4" />
      </Button>
    </div>
  )
} 