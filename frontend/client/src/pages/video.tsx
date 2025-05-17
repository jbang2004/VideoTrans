import { useState } from "react";
import { motion } from "framer-motion";
import { Folder, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import VideoModelCard from "@/components/video-model-card";
import { useLanguage } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";

export default function Video() {
  const [prompt, setPrompt] = useState("");
  const { language } = useLanguage();
  const { theme } = useTheme();

  const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(e.target.value);
  };

  const handleGenerate = () => {
    // Handle video generation
    console.log("Generating video with prompt:", prompt);
  };

  const placeholder = language === "zh" 
    ? "描述一个视频并点击生成..."
    : "Describe a video and click generate...";

  const startFrameLabel = language === "zh" ? "开始帧" : "Start frame";
  const endFrameLabel = language === "zh" ? "结束帧" : "End frame";
  const styleLabel = language === "zh" ? "风格" : "Style";
  const generateLabel = language === "zh" ? "生成" : "Generate";
  const modelsLabel = language === "zh" ? "模型" : "Models";

  const videoModels = [
    {
      name: "Wan 2.1",
      description: "New fast model with live previews",
      duration: "1 min.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
        </svg>
      ),
      features: ["Start frame"],
      bgColor: "bg-amber-100",
      textColor: "text-amber-500"
    },
    {
      name: "Hunyuan",
      description: "Fast and high-quality model",
      duration: "1 min.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
        </svg>
      ),
      bgColor: "bg-blue-100",
      textColor: "text-blue-500"
    },
    {
      name: "Kling 2.0",
      description: "New frontier model",
      duration: "5 min.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
        </svg>
      ),
      features: ["Start frame"],
      isExpensiveModel: true,
      bgColor: "bg-purple-100",
      textColor: "text-purple-500"
    }
  ];

  return (
    <motion.div 
      className="flex flex-col items-center justify-center"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="w-full max-w-2xl bg-muted p-6 rounded-xl">
        <Textarea 
          placeholder={placeholder} 
          className="w-full p-4 h-24 bg-card rounded-lg border border-border focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent resize-none"
          value={prompt}
          onChange={handlePromptChange}
        />
        
        <div className="flex flex-wrap gap-2 mt-4">
          <Button 
            variant="outline" 
            size="sm"
            className="px-3 py-1 rounded-md flex items-center text-sm transition-colors"
          >
            <Folder className="h-4 w-4 mr-1" />
            {startFrameLabel}
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            className="px-3 py-1 rounded-md flex items-center text-sm transition-colors"
          >
            <Folder className="h-4 w-4 mr-1" />
            {endFrameLabel}
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            className="px-3 py-1 rounded-md flex items-center text-sm transition-colors"
          >
            <Wand2 className="h-4 w-4 mr-1" />
            {styleLabel}
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            className="px-3 py-1 rounded-md text-sm transition-colors"
          >
            720p
          </Button>
        </div>
        
        <div className="flex justify-end mt-4">
          <Button 
            className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
            onClick={handleGenerate}
          >
            {generateLabel}
          </Button>
        </div>
      </div>
      
      <div className="w-full max-w-2xl mt-8">
        <h3 className="font-medium text-lg mb-4">{modelsLabel}</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {videoModels.map((model, index) => {
            const features = model.features?.map(feat => 
              feat === "Start frame" ? startFrameLabel : feat
            );
            
            return (
              <VideoModelCard
                key={index}
                name={model.name}
                description={language === "zh" ? 
                  (model.name === "Wan 2.1" ? "带实时预览的快速新模型" : 
                   model.name === "Hunyuan" ? "快速高质量模型" :
                   "新一代前沿模型") : 
                  model.description}
                duration={model.duration}
                icon={model.icon}
                features={features}
                isExpensiveModel={model.isExpensiveModel}
                bgColor={model.bgColor}
                textColor={model.textColor}
              />
            );
          })}
        </div>
      </div>
    </motion.div>
  );
}
