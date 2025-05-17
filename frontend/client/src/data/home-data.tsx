import { useLanguage } from "@/hooks/use-language";

export function useCarouselData() {
  const { language } = useLanguage();
  
  return [
    {
      id: "enhance-models",
      title: language === "zh" ? "全新增强模型" : "New Enhance Models",
      description: language === "zh" 
        ? "试用Krea强大的生成式增强器，使用Topaz将分辨率提升至22k，或者使用超快速模型节省积分。" 
        : "Try out Krea's new powerful generative enhancer, upscale to 22k resolution with Topaz, or save credits with super fast models.",
      buttonText: language === "zh" ? "提升与增强" : "Upscale & Enhance",
      buttonLink: "/enhancer",
      imageUrl: "https://images.unsplash.com/photo-1546182990-dffeafbe841d?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=800",
      tag: language === "zh" ? "新模型" : "New model"
    },
    {
      id: "visual-compositing",
      title: language === "zh" ? "使用ChatGPT Paint进行视觉合成" : "Visual Compositing with ChatGPT Paint",
      description: language === "zh" 
        ? "使用我们的新视觉画布构建图像，让ChatGPT解读您的草图、注释和拼贴。" 
        : "Build images with our new visual canvas & let ChatGPT interpret your sketches, annotations, and collages.",
      buttonText: language === "zh" ? "尝试ChatGPT Paint" : "Try ChatGPT Paint",
      buttonLink: "/image?gpt=1?new=true",
      imageUrl: "https://images.unsplash.com/photo-1561043433-aaf687c4cf04?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=800",
      tag: language === "zh" ? "新功能" : "New feature"
    },
    {
      id: "chatgpt-krea",
      title: language === "zh" ? "ChatGPT x Krea" : "ChatGPT x Krea",
      description: language === "zh" 
        ? "史上最强大的图像生成器。现在可在Krea Image、Krea Chat和Krea Stage中试用。" 
        : "The most powerful image generator ever. Try now in Krea Image, Krea Chat, and Krea Stage.",
      buttonText: language === "zh" ? "使用ChatGPT生成" : "Generate with ChatGPT",
      buttonLink: "/image?gpt=1",
      imageUrl: "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=800",
      tag: language === "zh" ? "新模型" : "New model"
    },
    {
      id: "video-restyle",
      title: language === "zh" ? "新品：视频重塑" : "New: Video Restyle",
      description: language === "zh" 
        ? "更改任何视频的风格。将朋友的视频转变为3D动画，克隆舞蹈，打造全新的视频风格。" 
        : "Change the style of any video. Turn videos of your friends into 3D animations, clone dances, craft totally new video styles.",
      buttonText: language === "zh" ? "重塑视频" : "Restyle Videos",
      buttonLink: "/video-restyle",
      imageUrl: "https://images.unsplash.com/photo-1536240478700-b869070f9279?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=800",
      tag: language === "zh" ? "新工具" : "New tool"
    },
    {
      id: "3d-objects",
      title: language === "zh" ? "推出3D对象" : "Introducing 3D Objects",
      description: language === "zh" 
        ? "几秒钟内从文字或图像生成3D对象。将照片或生成的资源转变为带纹理的3D网格。" 
        : "Generate 3D objects in seconds from text or images. Turn photos or generated assets into textured 3D meshes.",
      buttonText: language === "zh" ? "生成3D" : "Generate 3D",
      buttonLink: "/3d",
      imageUrl: "https://images.unsplash.com/photo-1638959383788-58f5d2afbd4c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=800",
      tag: language === "zh" ? "新工具" : "New Tool"
    }
  ];
}

export function useGenerateCardData() {
  const { language } = useLanguage();
  
  return [
    {
      title: language === "zh" ? "图像" : "Image",
      description: language === "zh" 
        ? "在Flux和Ideogram中生成自定义风格的图像。" 
        : "Generate images with custom styles in Flux and Ideogram.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      ),
      link: "/image",
      isNew: true,
      bgColor: "bg-blue-100",
      textColor: "text-blue-500"
    },
    {
      title: language === "zh" ? "视频" : "Video",
      description: language === "zh" 
        ? "使用Hailuo、Pika、Runway、Luma等生成视频。" 
        : "Generate videos with Hailuo, Pika, Runway, Luma, and more.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      ),
      link: "/video",
      bgColor: "bg-amber-100",
      textColor: "text-amber-500"
    },
    {
      title: language === "zh" ? "实时" : "Realtime",
      description: language === "zh" 
        ? "在画布上实时AI渲染。即时反馈循环。" 
        : "Realtime AI rendering on a canvas. Instant feedback loops.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
        </svg>
      ),
      link: "/realtime",
      bgColor: "bg-cyan-100",
      textColor: "text-cyan-500"
    },
    {
      title: language === "zh" ? "增强器" : "Enhancer",
      description: language === "zh" 
        ? "将真实和生成的图像提升和增强至4K。" 
        : "Upscale and enhance real and generated images up to 4K.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
      link: "/enhancer",
      isNew: true,
      bgColor: "bg-purple-100",
      textColor: "text-purple-500"
    }
  ];
}

export function useGalleryData() {
  const { language } = useLanguage();
  
  return [
    {
      id: "hulk-cat",
      title: language === "zh" ? "绿巨人猫咪怒吼" : "Hulk Cat Roars",
      image: "https://images.unsplash.com/photo-1526336024174-e58f5cdd8e13?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&h=600",
      link: "/feed/01968f5c-c249-7333-aeb6-80aa9517913f"
    },
    {
      id: "wizard-tower",
      title: language === "zh" ? "巫师塔内部" : "Wizard Tower Interior",
      image: "https://images.unsplash.com/photo-1518709414768-a88981a4515d?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&h=600",
      authorName: "plouz",
      link: "/feed/0196931d-a68c-7993-b952-d5551f73745f"
    },
    {
      id: "skull-man",
      title: language === "zh" ? "骷髅人" : "Skull Man",
      image: "https://images.unsplash.com/photo-1544161515-4ab6ce6db874?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&h=600",
      authorName: "frowenz",
      link: "/feed/01968975-522a-7449-b20d-d750307bd5fd"
    },
    {
      id: "heart-viz",
      title: language === "zh" ? "心脏逐渐增大" : "Heart Growing Larger",
      image: "https://pixabay.com/get/g8a69788e4f2c6371ef94e8622c61f3cdf10d531bb496d80fc3fefb15add9c49497089a811239a89365e73fb84cb76e7f38819dc18455e0c334a141dd05115861_1280.jpg",
      link: "/feed/01966d1b-0627-7117-bb34-2134a4a837b3"
    },
    {
      id: "emotional-dance",
      title: language === "zh" ? "角色情感舞蹈过渡" : "Character's Emotional Dance Transitions",
      image: "https://images.unsplash.com/photo-1517230878791-4d28214057c2?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&h=600",
      authorName: "plouz",
      link: "/feed/019670ea-80d0-7000-a777-af168e6ada19"
    },
    {
      id: "quiff-hair",
      title: language === "zh" ? "飞机头发型男士" : "Man with Quiff Hair",
      image: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&h=600",
      link: "/feed/01965bb1-741a-7eeb-a493-b145d9f6afa3"
    }
  ];
}