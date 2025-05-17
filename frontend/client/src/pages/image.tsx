import { useState } from "react";
import { motion } from "framer-motion";
import { Upload, User, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useLanguage } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";

export default function Image() {
  const [prompt, setPrompt] = useState("");
  const { t, language } = useLanguage();
  const { theme } = useTheme();

  const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(e.target.value);
  };

  const handleGenerate = () => {
    // Handle image generation
    console.log("Generating image with prompt:", prompt);
  };

  const placeholder = language === "zh" 
    ? "描述一个图像并点击生成..."
    : "Describe an image and click generate...";

  const uploadLabel = language === "zh" ? "上传图像" : "Upload image";
  const addPersonLabel = language === "zh" ? "添加人物" : "Add person";
  const applyStyleLabel = language === "zh" ? "应用样式" : "Apply style";
  const generateLabel = language === "zh" ? "生成" : "Generate";

  return (
    <motion.div 
      className="flex flex-col items-center justify-center min-h-[70vh]"
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
        
        <div className="flex justify-between mt-4">
          <div className="flex space-x-2">
            <Button 
              size="icon" 
              variant="outline" 
              className="w-8 h-8 rounded"
            >
              <Upload className="h-5 w-5" />
              <span className="sr-only">{uploadLabel}</span>
            </Button>
            <Button 
              size="icon" 
              variant="outline" 
              className="w-8 h-8 rounded"
            >
              <User className="h-5 w-5" />
              <span className="sr-only">{addPersonLabel}</span>
            </Button>
            <Button 
              size="icon" 
              variant="outline" 
              className="w-8 h-8 rounded"
            >
              <Wand2 className="h-5 w-5" />
              <span className="sr-only">{applyStyleLabel}</span>
            </Button>
          </div>
          
          <Button 
            className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
            onClick={handleGenerate}
          >
            {generateLabel}
          </Button>
        </div>
      </div>
    </motion.div>
  );
}
