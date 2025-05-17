import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { 
  addCircle, 
  folderOpen,
  mic,
  document,
  language,
  chevronDown
} from "ionicons/icons";
import { IonIcon } from "@ionic/react";
import { Button } from "@/components/ui/button";
import { useLanguage } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";
import { cn } from "@/lib/utils";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { BlurFade } from "@/components/magicui/blur-fade";

export default function AudioTranscription() {
  const { language: currentLanguage } = useLanguage();
  const { theme } = useTheme();
  const [selectedLanguage, setSelectedLanguage] = useState<string>(currentLanguage === "zh" ? "zh-CN" : "en-US");
  const [file, setFile] = useState<File | null>(null);
  const [transcriptionStarted, setTranscriptionStarted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [transcriptionResult, setTranscriptionResult] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 语言选项
  const languageOptions = [
    { value: "zh-CN", label: currentLanguage === "zh" ? "中文（简体）" : "Chinese (Simplified)" },
    { value: "en-US", label: currentLanguage === "zh" ? "英语（美国）" : "English (US)" },
    { value: "ja-JP", label: currentLanguage === "zh" ? "日语" : "Japanese" },
    { value: "ko-KR", label: currentLanguage === "zh" ? "韩语" : "Korean" },
    { value: "fr-FR", label: currentLanguage === "zh" ? "法语" : "French" },
    { value: "de-DE", label: currentLanguage === "zh" ? "德语" : "German" },
    { value: "es-ES", label: currentLanguage === "zh" ? "西班牙语" : "Spanish" },
  ];

  const handleFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setTranscriptionStarted(false);
      setTranscriptionResult([]);
    }
  };

  // 模拟转录过程
  const startTranscription = () => {
    if (!file) return;
    
    setTranscriptionStarted(true);
    setProgress(0);
    
    // 模拟进度条
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          // 模拟转录结果
          setTranscriptionResult([
            "嗨，谢谢你点开我们的视频，今天我要介绍我们公司的产品。",
            "我们的产品是基于人工智能的创新解决方案，可以帮助客户提高效率。",
            "我本人之前在科技行业工作了10年，对这个领域非常熟悉。",
            "好了，请问有什么问题想要了解的吗？有多长？多长时间？"
          ]);
          return 100;
        }
        return prev + 2;
      });
    }, 100);
  };

  // 根据语言选择显示的文本
  const title = currentLanguage === "zh" ? "音频转录" : "Audio Transcription";
  const description = currentLanguage === "zh" 
    ? "将语音自动转换为文本，支持多种语言的精准识别和转录" 
    : "Automatically convert speech to text with accurate recognition and transcription in multiple languages";
  const uploadLabel = currentLanguage === "zh" ? "上传音频" : "Upload";
  const selectLanguageLabel = currentLanguage === "zh" ? "选择语言" : "Select language";
  const transcribeText = currentLanguage === "zh" ? "开始转录" : "Start Transcription";
  const processingText = currentLanguage === "zh" ? "处理中..." : "Processing...";
  const transcriptionResultText = currentLanguage === "zh" ? "转录结果" : "Transcription Result";

  // 获取语言显示名称
  const getLanguageLabel = (value: string) => {
    const option = languageOptions.find(lang => lang.value === value);
    return option?.label || "";
  };

  return (
    <motion.div 
      className="flex flex-col items-center justify-center min-h-[70vh] py-12"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="w-full max-w-md mx-auto">
        <BlurFade
          layout
          delay={0.25}
          inView={true}
          className={cn(
            "p-6 rounded-3xl shadow-lg", 
            theme === "dark" ? "bg-zinc-900" : "bg-gray-100"
          )}
        >
          {/* 内容上部区域 - 缩小高度 */}
          <div className="h-[280px] mb-6 flex items-center justify-center">
            {/* 静态图片区域 - 苹果风格 */}
            <div className="relative w-full h-full overflow-hidden rounded-2xl bg-gradient-to-br from-indigo-100 to-indigo-300 flex items-center justify-center">
              {/* 音频相关图像 */}
              <div className="relative flex justify-center scale-110">
                <div className="absolute w-24 h-36 bg-indigo-500 rounded-lg transform -rotate-6 translate-x-6"></div>
                <div className="absolute w-24 h-36 bg-indigo-600 rounded-lg transform rotate-3 -translate-x-6"></div>
                <div className="absolute w-24 h-36 bg-indigo-400 rounded-lg transform rotate-0 z-10"></div>
                <div className="absolute inset-0 flex items-center justify-center z-20">
                  <IonIcon icon={mic} className="w-12 h-12 text-white" />
                </div>
              </div>
            </div>
          </div>
          
          {/* 标题和图标并排 */}
          <div className="flex items-center justify-center mb-3">
            <div className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center mr-3",
              theme === "dark" ? "bg-zinc-800" : "bg-indigo-100"
            )}>
              <IonIcon icon={mic} className="w-5 h-5" />
            </div>
            <h1 className="text-xl font-bold">{title}</h1>
          </div>
          
          {/* 描述文字 */}
          <p className="text-muted-foreground text-sm text-center mb-6">
            {description}
          </p>
          
          {/* 文件上传状态 */}
          {file && !transcriptionStarted && (
            <div className="mb-5 p-4 bg-background rounded-xl border border-border">
              <div className="flex items-center gap-3">
                <div className="bg-muted rounded-lg h-12 w-12 flex items-center justify-center flex-shrink-0">
                  <IonIcon icon={document} className="h-6 w-6 text-foreground/70" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium text-sm truncate">{file.name}</h3>
                  <p className="text-xs text-muted-foreground">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
                <Button 
                  className="bg-primary hover:bg-primary/90 text-white rounded-lg px-3 py-1 text-xs"
                  onClick={startTranscription}
                >
                  {transcribeText}
                </Button>
              </div>
            </div>
          )}
          
          {/* 转录进度 */}
          {transcriptionStarted && progress < 100 && (
            <div className="mb-5 p-4 bg-background rounded-xl border border-border">
              <div className="space-y-3">
                <Progress value={progress} className="h-2" />
                <p className="text-sm text-center text-muted-foreground">{processingText} {progress}%</p>
              </div>
            </div>
          )}
          
          {/* 转录结果 */}
          {transcriptionResult.length > 0 && (
            <div className="mb-5 p-4 bg-background rounded-xl border border-border">
              <h3 className="font-medium text-sm mb-3">{transcriptionResultText}</h3>
              <div className="max-h-36 overflow-y-auto space-y-3 text-xs">
                {transcriptionResult.map((line, index) => (
                  <div key={index} className="p-3 bg-muted rounded-lg">
                    {line}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* 按钮区域 - 左右排列 */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-5 w-full mt-6">
            <Button
              className={cn(
                "w-full sm:flex-1 h-14 text-white rounded-xl transition-colors flex items-center justify-center",
                "bg-indigo-600 hover:bg-indigo-700"
              )}
              onClick={handleFileClick}
            >
              <IonIcon icon={addCircle} className="w-6 h-6 mr-2" />
              <span className="text-base">{uploadLabel}</span>
            </Button>
            
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-full sm:flex-1 h-14 rounded-xl transition-colors flex items-center justify-center",
                    theme === "dark" ? "bg-zinc-800 border-zinc-700" : "bg-indigo-50 border-indigo-100 text-indigo-700"
                  )}
                >
                  <IonIcon icon={language} className="w-6 h-6 mr-2" />
                  <span className="text-base">{selectLanguageLabel}</span>
                  <IonIcon icon={chevronDown} className="w-4 h-4 ml-2" />
                </Button>
              </PopoverTrigger>
              <PopoverContent 
                className="w-64 p-3 rounded-xl"
                align="end"
              >
                <div className="space-y-1">
                  {languageOptions.map(lang => (
                    <Button
                      key={lang.value}
                      variant="ghost"
                      className={cn(
                        "w-full justify-start text-left font-normal",
                        selectedLanguage === lang.value && "bg-indigo-50 text-indigo-700 dark:bg-indigo-900/20 dark:text-indigo-500"
                      )}
                      onClick={() => setSelectedLanguage(lang.value)}
                    >
                      {lang.label}
                      {selectedLanguage === lang.value && (
                        <svg className="h-4 w-4 ml-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </Button>
                  ))}
                </div>
              </PopoverContent>
            </Popover>
          </div>
          
          <div className="text-center mt-4 text-xs text-muted-foreground">
            Max 20MB / MP3, WAV, M4A, FLAC
          </div>
        </BlurFade>
      </div>
      
      <input
        ref={fileInputRef}
        type="file"
        accept=".mp3,.wav,.m4a,.flac"
        className="hidden"
        onChange={handleFileChange}
      />
    </motion.div>
  );
}