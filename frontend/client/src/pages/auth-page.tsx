import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import { Mail, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { useLanguage } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";
import { useAuth } from "@/contexts/AuthContext";

export default function AuthPage() {
  const [, navigate] = useLocation();
  const [isRegistering, setIsRegistering] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const { toast } = useToast();
  const { t, language } = useLanguage();
  const { theme } = useTheme();
  const { user, signIn, signUp, loading, error: authError } = useAuth();

  useEffect(() => {
    if (user) {
      navigate("/");
    }
  }, [user, navigate]);

  const handleGoogleLogin = () => {
    toast({
      title: t("googleLoginNotAvailable"),
      description: t("featureNotImplemented") + " (Supabase)",
      variant: "destructive",
    });
  };

  const handleToggleMode = () => {
    setIsRegistering(!isRegistering);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!email.trim()) {
      toast({
        title: "Email required",
        description: "Please enter your email",
        variant: "destructive",
      });
      return;
    }
    if (!/\S+@\S+\.\S+/.test(email)) {
        toast({
            title: "Invalid email format",
            description: "Please enter a valid email address",
            variant: "destructive",
        });
        return;
    }

    if (!password) {
      toast({
        title: "Password required",
        description: "Please enter your password",
        variant: "destructive",
      });
      return;
    }
    if (password.length < 6) {
        toast({
            title: "Password too short",
            description: "Password should be at least 6 characters long",
            variant: "destructive",
        });
        return;
    }

    if (isRegistering && password !== confirmPassword) {
      toast({
        title: "Passwords do not match",
        description: "Please make sure your passwords match",
        variant: "destructive",
      });
      return;
    }

    let opError = null;
    if (isRegistering) {
      const { error } = await signUp({ email, password });
      opError = error;
      if (!error) {
        toast({
          title: "Registration Successful",
          description: "Please check your email for verification (if enabled), then log in.",
        });
        setIsRegistering(false);
        setEmail("");
        setPassword("");
        setConfirmPassword("");
      }
    } else {
      const { error } = await signIn({ email, password });
      opError = error;
      if (!error) {
        toast({
            title: "Login Successful",
        });
      } 
    }

    if (opError) {
      toast({
        title: isRegistering ? "Registration Failed" : "Login Failed",
        description: opError.message,
        variant: "destructive",
      });
    }
  };

  return (
    <div className="flex min-h-screen px-6 md:px-10 lg:px-16 py-8">
      <div className="flex w-full flex-col md:flex-row md:items-stretch max-w-6xl mx-auto gap-2 md:gap-4 self-center"> 
        {/* Auth Form Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full md:w-[45%]"
        >
          <div className={`p-4 h-full rounded-3xl ${theme === "dark" ? "bg-zinc-900" : "bg-gray-100"}`}>
            {/* 内容上部区域 - Krea Logo 与欢迎文字 */}
            <div className="h-[170px] mb-6 flex flex-col items-center justify-center">
              <div className="mb-4">
                <svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect width="56" height="56" rx="12" fill="black"/>
                  <path d="M28 21.5C28 18.4624 30.4624 16 33.5 16C36.5376 16 39 18.4624 39 21.5C39 24.5376 36.5376 27 33.5 27C30.4624 27 28 24.5376 28 21.5Z" fill="white"/>
                  <path d="M17 21.5C17 18.4624 19.4624 16 22.5 16C25.5376 16 28 18.4624 28 21.5C28 24.5376 25.5376 27 22.5 27C19.4624 27 17 24.5376 17 21.5Z" fill="white"/>
                  <path d="M28 34.5C28 31.4624 30.4624 29 33.5 29C36.5376 29 39 31.4624 39 34.5C39 37.5376 36.5376 40 33.5 40C30.4624 40 28 37.5376 28 34.5Z" fill="white"/>
                  <path d="M17 34.5C17 31.4624 19.4624 29 22.5 29C25.5376 29 28 31.4624 28 34.5C28 37.5376 25.5376 40 22.5 40C19.4624 40 17 37.5376 17 34.5Z" fill="white"/>
                </svg>
              </div>
              <h1 className="text-2xl font-semibold text-center">{t("welcomeToKrea")}</h1>
              <p className="text-sm text-muted-foreground mt-1">{t("loginOrSignup")}</p>
            </div>

            <div className="space-y-4">
              <Button 
                variant="outline" 
                className="w-full flex items-center justify-center gap-2 py-5 border-border rounded-xl"
                onClick={handleGoogleLogin}
              >
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M19.1 10.2C19.1 9.31 19.02 8.66 18.86 7.98H10V11.34H15.24C15.11 12.2 14.54 13.47 13.32 14.31L13.31 14.41L16.12 16.63L16.31 16.64C18.11 15 19.1 12.82 19.1 10.2Z" fill="#4285F4"/>
                  <path d="M10.0002 19.1C12.4302 19.1 14.4602 18.29 16.0002 16.82L13.3102 14.49C12.5202 15.04 11.4502 15.42 10.0002 15.42C7.5402 15.42 5.4602 13.8 4.7202 11.59L4.6302 11.59L1.7102 13.89L1.6602 13.98C3.1902 17.07 6.3602 19.1 10.0002 19.1Z" fill="#34A853"/>
                  <path d="M4.72 11.59C4.51 11.01 4.39 10.38 4.39 9.73C4.39 9.08 4.51 8.45 4.71 7.87L4.7 7.77L1.73 5.43L1.66 5.48C0.94 6.74 0.5 8.19 0.5 9.73C0.5 11.27 0.94 12.72 1.66 13.98L4.72 11.59Z" fill="#FBBC05"/>
                  <path d="M10.0002 4.04C11.7202 4.04 12.9402 4.74 13.6702 5.4L16.0702 3.08C14.4602 1.58 12.4302 0.73 10.0002 0.73C6.3602 0.73 3.1902 2.76 1.6602 5.85L4.7102 8.24C5.4602 6.03 7.5402 4.04 10.0002 4.04Z" fill="#EA4335"/>
                </svg>
                {t("continueWithGoogle")}
              </Button>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <Separator />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className={`${theme === 'dark' ? 'bg-zinc-900' : 'bg-gray-100'} px-2 text-muted-foreground`}>
                    {t("or")}
                  </span>
                </div>
              </div>

              <form onSubmit={handleSubmit}>
                <div className="space-y-4">
                  <div className="relative">
                    <Input
                      type="email"
                      placeholder={language === "zh" ? "邮箱" : "Email"}
                      className="pl-10 py-5"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      autoComplete="email"
                    />
                    <Mail className="h-5 w-5 absolute left-3 top-2.5 text-muted-foreground/60" />
                  </div>
                  
                  <div className="relative">
                    <Input
                      type="password"
                      placeholder={language === "zh" ? "密码" : "Password"}
                      className="pl-10 py-5"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      autoComplete={isRegistering ? "new-password" : "current-password"}
                    />
                    <Lock className="h-5 w-5 absolute left-3 top-2.5 text-muted-foreground/60" />
                  </div>

                  {isRegistering && (
                    <div className="relative">
                      <Input
                        type="password"
                        placeholder={language === "zh" ? "确认密码" : "Confirm Password"}
                        className="pl-10 py-5"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        autoComplete="new-password"
                      />
                      <Lock className="h-5 w-5 absolute left-3 top-2.5 text-muted-foreground/60" />
                    </div>
                  )}
                  
                  {authError && (
                    <p className="text-sm text-red-500 text-center">{authError.message}</p>
                  )}

                  <Button 
                    type="submit" 
                    className="w-full bg-primary hover:bg-primary/90 text-primary-foreground py-5 rounded-xl"
                    disabled={loading}
                  >
                    {loading 
                        ? (language === "zh" ? "处理中..." : "Processing...") 
                        : isRegistering 
                            ? (language === "zh" ? "注册" : "Sign Up") 
                            : (language === "zh" ? "登录" : "Login")}
                    {loading && (
                      <svg className="animate-spin ml-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                    )}
                  </Button>

                  <div className="text-center text-sm">
                    <span className="text-muted-foreground">
                      {isRegistering 
                        ? (language === "zh" ? "已有账户？" : "Already have an account?") 
                        : (language === "zh" ? "还没有账户？" : "Don't have an account?")}
                    </span>
                    <button
                      type="button"
                      className="text-primary ml-1 hover:underline"
                      onClick={handleToggleMode}
                    >
                      {isRegistering 
                        ? (language === "zh" ? "登录" : "Login") 
                        : (language === "zh" ? "注册" : "Sign up")}
                    </button>
                  </div>
                </div>
              </form>

              <p className="text-xs text-center text-muted-foreground mt-6">
                {t("bySigningUp")} <a href="#" className="text-primary hover:underline">{t("termsOfService")}</a> &{" "}
                <a href="#" className="text-primary hover:underline">{t("privacyPolicy")}</a>.
              </p>
            </div>
          </div>
        </motion.div>
        
        {/* Hero section - 蓝色卡片 */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="hidden md:block md:w-[55%]"
        >
          <div className="bg-blue-600 rounded-3xl text-white p-8 w-full h-full flex flex-col justify-center">
            <h2 className="text-3xl font-bold mb-4">{language === "zh" ? "创意无限，AI助力" : "Unlimited Creativity with AI"}</h2>
            <p className="text-lg opacity-90 mb-6">
              {language === "zh" 
                ? "使用我们强大的AI工具，轻松实现音频转录、文本转语音和视频翻译。" 
                : "Transform your content with our powerful AI tools for audio transcription, text-to-speech, and video translation."}
            </p>
            <ul className="space-y-4">
              <li className="flex items-center"><svg className="h-5 w-5 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>{language === "zh" ? "专业音频转文字" : "Professional Audio Transcription"}</li>
              <li className="flex items-center"><svg className="h-5 w-5 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>{language === "zh" ? "逼真的AI语音合成" : "Realistic AI Voice Synthesis"}</li>
              <li className="flex items-center"><svg className="h-5 w-5 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>{language === "zh" ? "视频字幕翻译" : "Video Subtitle Translation"}</li>
            </ul>
          </div>
        </motion.div>
      </div>
    </div>
  );
}