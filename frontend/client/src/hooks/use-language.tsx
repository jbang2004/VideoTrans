import { createContext, useContext, useEffect, useState, ReactNode } from "react";

// Define all our translation strings
export type TranslationKey = 
  // Auth page
  | "welcomeToKrea"
  | "loginOrSignup"
  | "continueWithGoogle"
  | "continueWithEmail"
  | "or"
  | "email"
  | "bySigningUp"
  | "termsOfService"
  | "privacyPolicy"
  | "googleLoginNotAvailable"
  | "emailLoginNotAvailable"
  | "featureNotImplemented"
  | "emailRequired"
  | "pleaseEnterEmail"
  
  // Navigation and common
  | "home"
  | "image"
  | "video"
  | "enhancer"
  | "edit"
  | "assets"
  | "pricing"
  | "logIn"
  | "signUp"
  | "logout"
  | "switchLanguage"
  | "english"
  | "chinese"
  | "pageNotFound"
  | "darkMode"
  | "lightMode"
  
  // New features
  | "audioTranscription"
  | "textToSpeech"
  | "videoTranslation"
  
  // Home page
  | "heroTitle"
  | "heroSubtitle"
  | "heroButtonText"
  | "featuredWorks"
  | "viewAll"
  | "createNow"
  | "getStarted"
  | "learnMore"
  
  // Image page
  | "imageGeneration"
  | "imageGenerationDesc"
  | "promptPlaceholder"
  | "generateImages"
  | "recentlyGenerated"
  | "styleOptions"
  
  // Video page
  | "videoGeneration"
  | "videoGenerationDesc"
  | "createVideo"
  | "videoModels"
  | "videoDuration"
  | "videoFeatures"
  
  // Enhancer page
  | "imageEnhancer"
  | "imageEnhancerDesc"
  | "uploadImage"
  | "dragAndDrop"
  | "enhanceNow"
  | "beforeAfter"
  
  // Pricing page
  | "kreaPlans"
  | "upgradeDesc"
  | "enterpriseAvailable"
  | "monthly"
  | "yearly"
  | "saveWithYearly"
  | "freePlan"
  | "basicPlan"
  | "proPlan"
  | "maxPlan"
  | "selectPlan"
  | "freeGenerations"
  | "limitedAccess"
  | "commercialLicense"
  | "faq"
  | "computeUnitsQuestion"
  | "computeUnitsAnswer"
  | "rolloverQuestion"
  | "rolloverAnswer"
  | "exceedLimitQuestion"
  | "exceedLimitAnswer"
  | "needHelp";

type Translations = {
  [key in TranslationKey]: string;
};

// Define the translations for English and Chinese
const translations: Record<"en" | "zh", Translations> = {
  en: {
    // Auth page
    welcomeToKrea: "Welcome to Krea",
    loginOrSignup: "Log in or sign up",
    continueWithGoogle: "Continue with Google",
    continueWithEmail: "Continue with email",
    or: "or",
    email: "Email",
    bySigningUp: "By signing up, you agree to our",
    termsOfService: "Terms of Service",
    privacyPolicy: "Privacy Policy",
    googleLoginNotAvailable: "Google login not available",
    emailLoginNotAvailable: "Email login not available",
    featureNotImplemented: "This feature is not implemented yet",
    emailRequired: "Email required",
    pleaseEnterEmail: "Please enter your email address",
    
    // Navigation and common
    home: "Home",
    image: "Image",
    video: "Video",
    enhancer: "Enhancer",
    edit: "Edit",
    assets: "Assets",
    pricing: "Pricing",
    logIn: "Log In",
    signUp: "Sign Up",
    logout: "Logout",
    switchLanguage: "Switch Language",
    english: "English",
    chinese: "Chinese",
    pageNotFound: "Page Not Found",
    darkMode: "Dark Mode",
    lightMode: "Light Mode",
    
    // New features
    audioTranscription: "Audio Transcription",
    textToSpeech: "Text to Speech",
    videoTranslation: "Video Translation",
    
    // Home page
    heroTitle: "AI-Powered Image & Video Creation",
    heroSubtitle: "Create stunning visuals with the power of AI. Generate, enhance, and transform your creative ideas into reality.",
    heroButtonText: "Try for Free",
    featuredWorks: "Featured Works",
    viewAll: "View All",
    createNow: "Create Now",
    getStarted: "Get Started",
    learnMore: "Learn More",
    
    // Image page
    imageGeneration: "AI Image Generation",
    imageGenerationDesc: "Create stunning images with simple text prompts. Our AI models will generate high-quality visuals based on your descriptions.",
    promptPlaceholder: "Describe the image you want to create...",
    generateImages: "Generate Images",
    recentlyGenerated: "Recently Generated",
    styleOptions: "Style Options",
    
    // Video page
    videoGeneration: "AI Video Creation",
    videoGenerationDesc: "Transform your ideas into high-quality videos with our advanced AI models.",
    createVideo: "Create Video",
    videoModels: "Video Models",
    videoDuration: "Duration",
    videoFeatures: "Features",
    
    // Enhancer page
    imageEnhancer: "Image Enhancer",
    imageEnhancerDesc: "Improve image quality, upscale resolution, and fix imperfections with our AI enhancement tools.",
    uploadImage: "Upload Image",
    dragAndDrop: "Drag and drop an image or click to browse",
    enhanceNow: "Enhance Now",
    beforeAfter: "Before / After",
    
    // Pricing page
    kreaPlans: "Krea Plans",
    upgradeDesc: "Upgrade to gain access to Pro features and generate more, faster.",
    enterpriseAvailable: "Enterprise and team plans now available",
    monthly: "Monthly",
    yearly: "Yearly",
    saveWithYearly: "Save with yearly",
    freePlan: "Free",
    basicPlan: "Basic",
    proPlan: "Pro",
    maxPlan: "Max",
    selectPlan: "Select Plan",
    freeGenerations: "Free daily generations",
    limitedAccess: "Limited access to KREA tools",
    commercialLicense: "Commercial license",
    faq: "Frequently Asked Questions",
    computeUnitsQuestion: "What are compute units?",
    computeUnitsAnswer: "Compute units are a measure of computational resources used to generate images or videos. They represent the processing power, memory, and time required for each creation.",
    rolloverQuestion: "Can I roll over unused compute units to the following month?",
    rolloverAnswer: "Compute units do not accumulate or carry over between billing cycles. At the start of each month, your compute unit balance is reset to your plan's allocated amount.",
    exceedLimitQuestion: "What options do I have if I exceed my compute unit limit?",
    exceedLimitAnswer: "If you exceed your compute unit limit, you can upgrade to a higher-tier subscription, purchase additional compute units, or utilize your daily free compute units.",
    needHelp: "Need help with your subscription? Reach out to"
  },
  zh: {
    // Auth page
    welcomeToKrea: "欢迎来到 Krea",
    loginOrSignup: "登录或注册",
    continueWithGoogle: "使用谷歌账号继续",
    continueWithEmail: "使用邮箱继续",
    or: "或",
    email: "邮箱",
    bySigningUp: "注册即表示您同意我们的",
    termsOfService: "服务条款",
    privacyPolicy: "隐私政策",
    googleLoginNotAvailable: "谷歌登录不可用",
    emailLoginNotAvailable: "邮箱登录不可用",
    featureNotImplemented: "此功能尚未实现",
    emailRequired: "需要邮箱",
    pleaseEnterEmail: "请输入您的邮箱地址",
    
    // Navigation and common
    home: "首页",
    image: "图片",
    video: "视频",
    enhancer: "增强器",
    edit: "编辑",
    assets: "资源",
    pricing: "价格",
    logIn: "登录",
    signUp: "注册",
    logout: "退出",
    switchLanguage: "切换语言",
    english: "英文",
    chinese: "中文",
    pageNotFound: "页面未找到",
    darkMode: "深色模式",
    lightMode: "浅色模式",
    
    // New features
    audioTranscription: "音频转录",
    textToSpeech: "文本配音",
    videoTranslation: "视频翻译",
    
    // Home page
    heroTitle: "AI驱动的图像和视频创作",
    heroSubtitle: "利用AI的力量创建令人惊叹的视觉效果。生成、增强和将您的创意转化为现实。",
    heroButtonText: "免费试用",
    featuredWorks: "精选作品",
    viewAll: "查看全部",
    createNow: "立即创建",
    getStarted: "开始使用",
    learnMore: "了解更多",
    
    // Image page
    imageGeneration: "AI图像生成",
    imageGenerationDesc: "通过简单的文字提示创建令人惊叹的图像。我们的AI模型将根据您的描述生成高质量的视觉效果。",
    promptPlaceholder: "描述您想要创建的图像...",
    generateImages: "生成图像",
    recentlyGenerated: "最近生成",
    styleOptions: "风格选项",
    
    // Video page
    videoGeneration: "AI视频创作",
    videoGenerationDesc: "利用我们先进的AI模型将您的想法转变为高质量视频。",
    createVideo: "创建视频",
    videoModels: "视频模型",
    videoDuration: "时长",
    videoFeatures: "特点",
    
    // Enhancer page
    imageEnhancer: "图像增强器",
    imageEnhancerDesc: "利用我们的AI增强工具提高图像质量，提升分辨率并修复瑕疵。",
    uploadImage: "上传图像",
    dragAndDrop: "拖放图像或点击浏览",
    enhanceNow: "立即增强",
    beforeAfter: "对比前后",
    
    // Pricing page
    kreaPlans: "Krea套餐",
    upgradeDesc: "升级以获取专业功能，更快生成更多内容。",
    enterpriseAvailable: "现已提供企业和团队方案",
    monthly: "月付",
    yearly: "年付",
    saveWithYearly: "年付更省钱",
    freePlan: "免费",
    basicPlan: "基础",
    proPlan: "专业",
    maxPlan: "高级",
    selectPlan: "选择套餐",
    freeGenerations: "每日免费生成",
    limitedAccess: "有限访问KREA工具",
    commercialLicense: "商业许可",
    faq: "常见问题",
    computeUnitsQuestion: "什么是计算单元？",
    computeUnitsAnswer: "计算单元是衡量生成图像或视频所用计算资源的指标。它们代表每次创作所需的处理能力、内存和时间。",
    rolloverQuestion: "我可以将未用完的计算单元结转到下个月吗？",
    rolloverAnswer: "计算单元不会在计费周期之间累积或结转。每个月初，您的计算单元余额将重置为您套餐分配的数量。",
    exceedLimitQuestion: "如果我超出计算单元限制有哪些选择？",
    exceedLimitAnswer: "如果您超出计算单元限制，可以升级到更高级别的订阅，购买额外的计算单元，或利用每日免费计算单元。",
    needHelp: "需要订阅帮助？请联系"
  },
};

type LanguageContextType = {
  language: "en" | "zh";
  setLanguage: (language: "en" | "zh") => void;
  t: (key: TranslationKey) => string;
};

const LanguageContext = createContext<LanguageContextType | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<"en" | "zh">("en");

  useEffect(() => {
    // Get the saved language preference from localStorage
    const savedLanguage = localStorage.getItem("language") as "en" | "zh";
    if (savedLanguage && (savedLanguage === "en" || savedLanguage === "zh")) {
      setLanguage(savedLanguage);
    }
  }, []);

  const changeLanguage = (newLanguage: "en" | "zh") => {
    setLanguage(newLanguage);
    // Save the language preference to localStorage
    localStorage.setItem("language", newLanguage);
  };

  // Function to get a translated string by key
  const t = (key: TranslationKey): string => {
    return translations[language][key] || key;
  };

  return (
    <LanguageContext.Provider
      value={{
        language,
        setLanguage: changeLanguage,
        t,
      }}
    >
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return context;
}