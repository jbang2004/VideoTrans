import { useState } from "react";
import { motion } from "framer-motion";
import { 
  addCircle, 
  volumeHigh,
  person,
  settings,
  document,
  play,
  download,
  chevronDown
} from "ionicons/icons";
import { IonIcon } from "@ionic/react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { useLanguage } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";
import { cn } from "@/lib/utils";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { BlurFade } from "@/components/magicui/blur-fade";

interface Voice {
  id: string;
  name: string;
  language: string;
  gender: string;
  avatar?: string;
}

export default function TextToSpeech() {
  const { language: currentLanguage } = useLanguage();
  const { theme } = useTheme();
  const [text, setText] = useState<string>("");
  const [selectedVoice, setSelectedVoice] = useState<string | null>(null);
  const [speed, setSpeed] = useState<number[]>([1]);
  const [pitch, setPitch] = useState<number[]>([1]);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState<boolean>(false);

  // 示例的语音列表
  const voices: Voice[] = [
    { id: "voice1", name: "王小明", language: "zh-CN", gender: "male", avatar: "https://api.dicebear.com/7.x/thumbs/svg?seed=Felix&eyesColor=0a0a0a" },
    { id: "voice2", name: "李小花", language: "zh-CN", gender: "female", avatar: "https://api.dicebear.com/7.x/thumbs/svg?seed=Lilly&eyesColor=0a0a0a" },
    { id: "voice3", name: "John", language: "en-US", gender: "male", avatar: "https://api.dicebear.com/7.x/thumbs/svg?seed=John&eyesColor=0a0a0a" },
    { id: "voice4", name: "Sarah", language: "en-US", gender: "female", avatar: "https://api.dicebear.com/7.x/thumbs/svg?seed=Sarah&eyesColor=0a0a0a" },
    { id: "voice5", name: "智能女声", language: "zh-CN", gender: "female", avatar: "https://api.dicebear.com/7.x/thumbs/svg?seed=AI1&eyesColor=0a0a0a" },
    { id: "voice6", name: "智能男声", language: "zh-CN", gender: "male", avatar: "https://api.dicebear.com/7.x/thumbs/svg?seed=AI2&eyesColor=0a0a0a" },
  ];

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    // 如果有生成的音频，当文本改变时重置
    if (generatedAudioUrl) {
      setGeneratedAudioUrl(null);
      setIsPlaying(false);
    }
  };

  const generateSpeech = () => {
    if (text.trim() === "" || !selectedVoice) return;
    
    setIsGenerating(true);
    
    // 模拟生成过程
    setTimeout(() => {
      // 这里只是模拟URL，实际应用中会从API获取真实的音频URL
      setGeneratedAudioUrl("https://example.com/generated-audio.mp3");
      setIsGenerating(false);
      setIsPlaying(true);
    }, 2000);
  };

  const togglePlayback = () => {
    if (generatedAudioUrl) {
      setIsPlaying(!isPlaying);
    }
  };

  // 根据语言选择显示的文本
  const title = currentLanguage === "zh" ? "文本配音" : "Text to Speech";
  const description = currentLanguage === "zh" 
    ? "将文本转换为逼真的语音，提供多种声音和风格选择" 
    : "Convert text to realistic speech with multiple voices and styles";
  const textPlaceholder = currentLanguage === "zh" 
    ? "在此输入您想要转换成语音的文字..." 
    : "Enter text you want to convert to speech here...";
  const generateButtonText = currentLanguage === "zh" ? "生成语音" : "Generate";
  const processingText = currentLanguage === "zh" ? "处理中..." : "Processing...";
  const selectVoiceText = currentLanguage === "zh" ? "选择声音" : "Select Voice";
  const speedText = currentLanguage === "zh" ? "速度" : "Speed";
  const pitchText = currentLanguage === "zh" ? "音高" : "Pitch";
  const characterCountText = currentLanguage === "zh" ? "字符数" : "Characters";
  const settingsText = currentLanguage === "zh" ? "设置" : "Settings";

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
            <div className="relative w-full h-full overflow-hidden rounded-2xl bg-gradient-to-br from-purple-100 to-purple-300 flex items-center justify-center">
              {/* 语音相关图像 */}
              <div className="relative flex justify-center scale-110">
                <div className="absolute w-24 h-36 bg-purple-500 rounded-lg transform -rotate-6 translate-x-6"></div>
                <div className="absolute w-24 h-36 bg-purple-600 rounded-lg transform rotate-3 -translate-x-6"></div>
                <div className="absolute w-24 h-36 bg-purple-400 rounded-lg transform rotate-0 z-10"></div>
                <div className="absolute inset-0 flex items-center justify-center z-20">
                  <IonIcon icon={volumeHigh} className="w-12 h-12 text-white" />
                </div>
              </div>
            </div>
          </div>
          
          {/* 标题和图标并排 */}
          <div className="flex items-center justify-center mb-3">
            <div className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center mr-3",
              theme === "dark" ? "bg-zinc-800" : "bg-purple-100"
            )}>
              <IonIcon icon={volumeHigh} className="w-5 h-5" />
            </div>
            <h1 className="text-xl font-bold">{title}</h1>
          </div>
          
          {/* 描述文字 */}
          <p className="text-muted-foreground text-sm text-center mb-6">
            {description}
          </p>
          
          {/* 文本输入区域 */}
          <div className="mb-5">
            <Textarea 
              value={text}
              onChange={handleTextChange}
              placeholder={textPlaceholder}
              className="w-full p-4 h-28 bg-background rounded-xl border border-border focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent resize-none text-sm"
            />
            <div className="flex justify-between mt-2 items-center">
              <div className="text-xs text-muted-foreground">
                {characterCountText}: {text.length}
              </div>
            </div>
          </div>
          
          {/* 选择声音部分 */}
          {selectedVoice && (
            <div className="mb-5 p-4 bg-background rounded-xl border border-border">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full overflow-hidden bg-muted">
                    {voices.find(v => v.id === selectedVoice)?.avatar ? (
                      <img 
                        src={voices.find(v => v.id === selectedVoice)?.avatar} 
                        alt={voices.find(v => v.id === selectedVoice)?.name} 
                        className="w-full h-full object-cover" 
                      />
                    ) : (
                      <IonIcon icon={person} className="w-full h-full p-2" />
                    )}
                  </div>
                  <div>
                    <h3 className="text-sm font-medium">
                      {voices.find(v => v.id === selectedVoice)?.name}
                    </h3>
                    <p className="text-xs text-muted-foreground">
                      {voices.find(v => v.id === selectedVoice)?.language}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-10 w-10"
                  onClick={() => setShowSettings(!showSettings)}
                >
                  <IonIcon icon={settings} className="h-5 w-5" />
                </Button>
              </div>
              
              {/* 语音设置 */}
              {showSettings && (
                <div className="mt-4 pt-4 border-t border-border space-y-4">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-xs">{speedText}</label>
                      <span className="text-xs bg-muted px-2 py-1 rounded text-muted-foreground">
                        {speed[0].toFixed(1)}x
                      </span>
                    </div>
                    <Slider
                      value={speed}
                      min={0.5}
                      max={2}
                      step={0.1}
                      onValueChange={setSpeed}
                      className="h-4"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-xs">{pitchText}</label>
                      <span className="text-xs bg-muted px-2 py-1 rounded text-muted-foreground">
                        {pitch[0].toFixed(1)}
                      </span>
                    </div>
                    <Slider
                      value={pitch}
                      min={0.5}
                      max={2}
                      step={0.1}
                      onValueChange={setPitch}
                      className="h-4"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* 生成的音频显示 */}
          {generatedAudioUrl && (
            <div className="mb-5 p-4 bg-background rounded-xl border border-border">
              <div className="flex items-center justify-between">
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-12 w-12 text-primary"
                  onClick={togglePlayback}
                >
                  <IonIcon icon={play} className="h-7 w-7" />
                </Button>
                <div className="flex-1 mx-3 h-3 bg-muted rounded-full">
                  <div 
                    className={`h-full bg-primary rounded-full transition-all duration-200 ${
                      isPlaying ? "w-1/2" : "w-0"
                    }`} 
                  />
                </div>
                <Button variant="ghost" size="icon" className="h-10 w-10">
                  <IonIcon icon={download} className="h-5 w-5" />
                </Button>
              </div>
            </div>
          )}
          
          {/* 按钮区域 - 左右排列 */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-5 w-full mt-6">
            <Button
              className={cn(
                "w-full sm:flex-1 h-14 text-white rounded-xl transition-colors flex items-center justify-center",
                "bg-purple-600 hover:bg-purple-700"
              )}
              onClick={generateSpeech}
              disabled={text.trim() === "" || isGenerating || !selectedVoice}
            >
              <IonIcon icon={addCircle} className="w-6 h-6 mr-2" />
              <span className="text-base">{isGenerating ? processingText : generateButtonText}</span>
            </Button>
            
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-full sm:flex-1 h-14 rounded-xl transition-colors flex items-center justify-center",
                    theme === "dark" ? "bg-zinc-800 border-zinc-700" : "bg-purple-50 border-purple-100 text-purple-700"
                  )}
                >
                  <IonIcon icon={person} className="w-6 h-6 mr-2" />
                  <span className="text-base">{selectVoiceText}</span>
                  <IonIcon icon={chevronDown} className="w-4 h-4 ml-2" />
                </Button>
              </PopoverTrigger>
              <PopoverContent 
                className="w-64 p-3 rounded-xl"
                align="end"
              >
                <div className="space-y-1 max-h-64 overflow-y-auto">
                  {voices.map(voice => (
                    <Button
                      key={voice.id}
                      variant="ghost"
                      className={cn(
                        "w-full justify-start text-left font-normal",
                        selectedVoice === voice.id && "bg-purple-50 text-purple-700 dark:bg-purple-900/20 dark:text-purple-500"
                      )}
                      onClick={() => setSelectedVoice(voice.id)}
                    >
                      <div className="flex items-center w-full">
                        <div className="w-8 h-8 rounded-full overflow-hidden bg-muted mr-2 flex-shrink-0">
                          {voice.avatar ? (
                            <img src={voice.avatar} alt={voice.name} className="w-full h-full object-cover" />
                          ) : (
                            <IonIcon icon={person} className="w-full h-full p-1.5" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm truncate">{voice.name}</p>
                          <p className="text-xs text-muted-foreground truncate">{voice.language}</p>
                        </div>
                        {selectedVoice === voice.id && (
                          <svg className="h-4 w-4 ml-auto flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                      </div>
                    </Button>
                  ))}
                </div>
              </PopoverContent>
            </Popover>
          </div>
          
          <div className="text-center mt-4 text-xs text-muted-foreground">
            {currentLanguage === "zh" ? "支持40种语言，100+种声音" : "Supports 40 languages, 100+ voices"}
          </div>
        </BlurFade>
      </div>
    </motion.div>
  );
}