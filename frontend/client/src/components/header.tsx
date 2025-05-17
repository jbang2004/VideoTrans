import { useState } from "react";
import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";
import { useMobile } from "@/hooks/use-mobile";
import { useLanguage, TranslationKey } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";
import { useAuth } from "@/contexts/AuthContext";
import { 
  home, 
  mic, 
  volumeHigh, 
  videocam, 
  image, 
  colorWand, 
  sunny, 
  moon, 
  globe, 
  globeOutline,
  menu, 
  close, 
  person, 
  wallet,
  logOut
} from "ionicons/icons";
import { IonIcon } from "@ionic/react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

type NavItem = {
  path: string;
  labelKey: TranslationKey;
  icon: string;
};

const NavItems: NavItem[] = [
  {
    path: "/",
    labelKey: "home",
    icon: home,
  },
  {
    path: "/audio-transcription",
    labelKey: "audioTranscription",
    icon: mic,
  },
  {
    path: "/text-to-speech",
    labelKey: "textToSpeech",
    icon: volumeHigh,
  },
  {
    path: "/video-translation",
    labelKey: "videoTranslation",
    icon: videocam,
  },
];

export default function Header() {
  const [isOpen, setIsOpen] = useState(false);
  const [location] = useLocation();
  const isMobile = useMobile();
  const { t, language: currentLanguage, setLanguage } = useLanguage();
  const { theme, toggleTheme } = useTheme();
  const { user, signOut, loading: authLoading } = useAuth();

  const toggleMenu = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);
  
  const handleLogout = async () => {
    await signOut();
  };

  return (
    <header className="sticky top-0 z-50 bg-transparent w-full py-4">
      <div className="w-full px-2 sm:px-4 md:px-6 flex items-center justify-between h-12">
        {/* Logo */}
        <div className="flex items-center">
          <Link 
            href="/" 
            onClick={closeMenu} 
            className="flex items-center justify-center h-11 w-11 bg-white/60 dark:bg-gray-800/60 rounded-xl shadow-sm hover:bg-white/70 dark:hover:bg-gray-700/70 transition-colors backdrop-blur-md"
          >
            <svg className="w-7 h-7" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect width="24" height="24" rx="4" fill={theme === 'dark' ? '#FFFFFF' : '#000000'} />
              <path d="M12 9.5C12 8.67157 12.6716 8 13.5 8C14.3284 8 15 8.67157 15 9.5C15 10.3284 14.3284 11 13.5 11C12.6716 11 12 10.3284 12 9.5Z" fill={theme === 'dark' ? '#000000' : '#FFFFFF'} />
              <path d="M9 9.5C9 8.67157 9.67157 8 10.5 8C11.3284 8 12 8.67157 12 9.5C12 10.3284 11.3284 11 10.5 11C9.67157 11 9 10.3284 9 9.5Z" fill={theme === 'dark' ? '#000000' : '#FFFFFF'} />
              <path d="M12 13.5C12 12.6716 12.6716 12 13.5 12C14.3284 12 15 12.6716 15 13.5C15 14.3284 14.3284 15 13.5 15C12.6716 15 12 14.3284 12 13.5Z" fill={theme === 'dark' ? '#000000' : '#FFFFFF'} />
              <path d="M9 13.5C9 12.6716 9.67157 12 10.5 12C11.3284 12 12 12.6716 12 13.5C12 14.3284 11.3284 15 10.5 15C9.67157 15 9 14.3284 9 13.5Z" fill={theme === 'dark' ? '#000000' : '#FFFFFF'} />
            </svg>
          </Link>
        </div>

        {/* Navigation - Floating center menu like Krea.ai */}
        <div 
          className="hidden md:flex items-center bg-gray-100/90 dark:bg-gray-800/80 rounded-2xl px-3 py-2 absolute left-1/2 transform -translate-x-1/2 shadow-sm backdrop-blur-md"
        >
          {NavItems.map((item, index) => (
            <div 
              key={item.path} 
              className={cn(
                "relative group",
                index === 0 ? "ml-0" : "ml-1",
                index === NavItems.length - 1 ? "mr-0" : "mr-1"
              )}
            >
              <Link 
                href={item.path}
                onClick={closeMenu}
                className={cn(
                  "flex items-center justify-center transition-colors",
                  location === item.path 
                    ? "bg-white dark:bg-gray-700 shadow-sm px-5 py-2.5 rounded-xl" 
                    : "hover:bg-white/70 dark:hover:bg-gray-700/70 px-5 py-2.5 rounded-xl"
                )}
                aria-label={t(item.labelKey)}
              >
                <IonIcon icon={item.icon} className="h-5 w-5" />
              </Link>
              {/* 悬浮时显示的标题 */}
              <div className="absolute -bottom-10 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity px-3 py-1 bg-gray-200/90 dark:bg-gray-600/90 text-xs font-medium rounded-lg backdrop-blur-md shadow-sm whitespace-nowrap z-10">
                {t(item.labelKey)}
              </div>
            </div>
          ))}
        </div>
        
        {/* Mobile menu button */}
        {isMobile && (
          <button 
            className="h-11 w-11 ml-2 flex items-center justify-center rounded-xl bg-white/60 dark:bg-gray-800/60 hover:bg-white/70 dark:hover:bg-gray-700/70 transition-colors md:hidden shadow-sm backdrop-blur-md"
            onClick={toggleMenu}
            aria-label="Toggle menu"
            disabled={authLoading}
          >
            <IonIcon icon={isOpen ? close : menu} className="h-6 w-6" />
          </button>
        )}

        {/* Right side menu - Floating buttons like Krea.ai */}
        <div className="flex items-center space-x-4">
          <button
            onClick={toggleTheme}
            className="hidden xl:flex h-11 w-11 items-center justify-center rounded-xl bg-white/60 dark:bg-gray-800/60 hover:bg-white/70 dark:hover:bg-gray-700/70 transition-colors shadow-sm backdrop-blur-md"
            aria-label={theme === "dark" ? t("lightMode") : t("darkMode")}
          >
            <IonIcon icon={theme === "dark" ? sunny : moon} className="h-4.5 w-4.5" />
          </button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button 
                className="hidden lg:flex h-11 w-11 items-center justify-center rounded-xl bg-white/60 dark:bg-gray-800/60 hover:bg-white/70 dark:hover:bg-gray-700/70 transition-colors shadow-sm backdrop-blur-md"
                aria-label={t("switchLanguage")}
              >
                <IonIcon icon={globeOutline} className="h-4.5 w-4.5" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setLanguage("en")}>
                <span className={cn(currentLanguage === "en" && "font-bold")}>
                  {t("english")}
                </span>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setLanguage("zh")}>
                <span className={cn(currentLanguage === "zh" && "font-bold")}>
                  {t("chinese")}
                </span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Floating button for Pricing */}
          <Link 
            href="/pricing" 
            className="hidden lg:block py-2.5 px-6 rounded-xl bg-gray-200/60 dark:bg-gray-700/60 text-sm font-medium hover:bg-gray-300/70 dark:hover:bg-gray-600/70 transition-colors shadow-sm backdrop-blur-md"
          >
            {t("pricing")}
          </Link>
          
          {/* 用户未登录时显示登录/注册按钮，已登录时显示退出按钮 */}
          {authLoading ? (
            <div className="h-8 w-20 bg-gray-200 dark:bg-gray-700 animate-pulse rounded-xl"></div>
          ) : user ? (
            <button
              onClick={handleLogout}
              className="hidden md:flex items-center py-2.5 px-3 sm:px-4 md:px-6 rounded-xl bg-red-600/90 hover:bg-red-700/90 text-white text-xs sm:text-sm font-medium transition-colors shadow-sm backdrop-blur-md"
              disabled={authLoading}
            >
              <IonIcon icon={logOut} className="h-5 w-5 mr-1" />
              <span>{t("logout")}</span>
            </button>
          ) : (
            <Link 
              href="/auth" 
              className="hidden md:block py-2.5 px-3 sm:px-4 md:px-6 rounded-xl bg-blue-600/90 hover:bg-blue-700/90 text-white text-xs sm:text-sm font-medium transition-colors shadow-sm backdrop-blur-md"
            >
              {t("signUp")}
            </Link>
          )}
        </div>
      </div>
      
      {/* Mobile Menu */}
      {isMobile && isOpen && (
        <div className="md:hidden absolute top-20 left-2 right-2 sm:left-4 sm:right-4 bg-white/70 dark:bg-gray-800/70 backdrop-blur-xl rounded-xl shadow-lg p-3 sm:p-4 space-y-2 z-50">
          {NavItems.map((item) => (
            <Link 
              key={item.path} 
              href={item.path}
              onClick={closeMenu}
              className={cn(
                "flex items-center space-x-2 p-2 transition-colors",
                location === item.path 
                  ? "bg-white/80 dark:bg-gray-700/80 backdrop-blur-sm rounded-xl px-4" 
                  : "hover:bg-white/50 dark:hover:bg-gray-700/50 rounded-xl"
              )}
            >
              <span className="w-8 h-8 flex items-center justify-center">
                <IonIcon icon={item.icon} className="h-5 w-5" />
              </span>
              <span className="font-medium">{t(item.labelKey)}</span>
            </Link>
          ))}
          <div className="border-t border-white/20 dark:border-gray-700/50 pt-2 mt-2 flex flex-col space-y-2">
            <Link 
              href="/pricing" 
              onClick={closeMenu} 
              className="flex items-center p-2 rounded-xl hover:bg-white/50 dark:hover:bg-gray-700/50 transition-colors"
            >
              <span className="w-8 h-8 flex items-center justify-center">
                <IonIcon icon={wallet} className="h-5 w-5" />
              </span>
              <span className="font-medium">{t("pricing")}</span>
            </Link>
            
            {authLoading ? (
                <div className="h-10 bg-gray-200 dark:bg-gray-700 animate-pulse rounded-xl mx-2"></div>
            ) : user ? (
              <button
                onClick={() => {
                  handleLogout();
                  closeMenu();
                }}
                className="flex items-center p-2 rounded-xl hover:bg-white/50 dark:hover:bg-gray-700/50 transition-colors text-red-600"
              >
                <span className="w-8 h-8 flex items-center justify-center">
                  <IonIcon icon={logOut} className="h-5 w-5" />
                </span>
                <span className="font-medium">{t("logout")}</span>
              </button>
            ) : (
              <Link 
                href="/auth"
                onClick={closeMenu}
                className="flex items-center p-2 rounded-xl hover:bg-white/50 dark:hover:bg-gray-700/50 transition-colors"
              >
                <span className="w-8 h-8 flex items-center justify-center">
                  <IonIcon icon={person} className="h-5 w-5" />
                </span>
                <span className="font-medium">{t("signUp")}</span>
              </Link>
            )}
          </div>
        </div>
      )}
    </header>
  );
}