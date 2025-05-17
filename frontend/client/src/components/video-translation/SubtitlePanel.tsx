import { motion } from "framer-motion";
import {
  time,
  caretForward,
  pencil,
  checkmark,
  close,
  chevronDown,
  language as languageIcon
} from "ionicons/icons";
import { IonIcon } from "@ionic/react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { Subtitle } from "@/types";
import { Translations } from "@/lib/translations";

interface SubtitlesPanelProps {
  theme: string | undefined;
  subtitles: Subtitle[];
  editingSubtitleId: string | null;
  targetLanguage: string;
  translations: Translations;
  isMobile: boolean;
  isLoading: boolean;
  error: string | null;
  getLanguageLabel: (value: string) => string;
  jumpToTime: (timeString: string) => void;
  updateSubtitleTranslation: (id: string, newTranslation: string) => void;
  toggleEditMode: (id: string) => void;
  setTargetLanguage: (language: string) => void;
  fetchSubtitles: (taskId: string, targetLang: string) => Promise<void>;
  closeSubtitlesPanel: () => void;
  subtitlesContainerRef: React.RefObject<HTMLDivElement>;
  currentTaskId: string;
}

export default function SubtitlesPanel({
  theme,
  subtitles,
  editingSubtitleId,
  targetLanguage,
  translations: T,
  isMobile,
  isLoading,
  error,
  getLanguageLabel,
  jumpToTime,
  updateSubtitleTranslation,
  toggleEditMode,
  setTargetLanguage,
  fetchSubtitles,
  closeSubtitlesPanel,
  subtitlesContainerRef,
  currentTaskId
}: SubtitlesPanelProps) {

  const handleFetchSubtitles = () => {
    if (currentTaskId && targetLanguage) {
      fetchSubtitles(currentTaskId, targetLanguage);
    }
  };

  return (
    <div className={cn(
      "p-2 sm:p-3 rounded-3xl relative flex flex-col",
      "bg-transparent"
    )}>
      <Button
        variant="ghost"
        size="sm"
        className="absolute top-3 right-3 h-8 w-8 rounded-full p-0 z-20"
        onClick={closeSubtitlesPanel}
      >
        <IonIcon icon={close} className="h-4 w-4" />
        <span className="sr-only">{T.closeLabel}</span>
      </Button>
      
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 pr-10">
        <h2 className="text-xl font-bold mb-2 sm:mb-0 whitespace-nowrap">{T.translatedSubtitleLabel}</h2>
        <div className="flex items-center space-x-2 flex-wrap sm:flex-nowrap mt-2 sm:mt-0">
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 px-2 text-xs whitespace-nowrap">
                {T.languageLabel || "Language"}: {getLanguageLabel(targetLanguage)}
                <IonIcon icon={chevronDown} className="ml-1 h-3 w-3" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-40 p-0" align="end">
              <div className="p-1">
                {T.languageOptions.map((lang: { value: string; label: string }) => (
                  <Button
                    key={lang.value}
                    variant="ghost"
                    size="sm"
                    className={cn(
                      "w-full justify-start text-left text-xs",
                      targetLanguage === lang.value && "bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-500"
                    )}
                    onClick={() => setTargetLanguage(lang.value)}
                  >
                    {lang.label}
                  </Button>
                ))}
              </div>
            </PopoverContent>
          </Popover>
          <Button 
            variant="default" 
            size="sm" 
            className="h-8 px-3 text-xs bg-blue-600 hover:bg-blue-700 text-white whitespace-nowrap"
            onClick={handleFetchSubtitles}
            disabled={isLoading || !targetLanguage || !currentTaskId}
          >
            <IonIcon icon={languageIcon} className="h-4 w-4 mr-1.5" />
            {isLoading ? (T.loadingLabel || "Loading...") : (T.translateButtonLabel || "Translate")}
          </Button>
        </div>
      </div>
      
      <div className="h-[500px] overflow-hidden [mask-image:linear-gradient(to_bottom,transparent,black_10%,black_90%,transparent)]">
        <ScrollArea className={cn("rounded-md h-full")}>
          <div className="space-y-4 pb-4" ref={subtitlesContainerRef}>
            {isLoading && (
              <div className="flex justify-center items-center h-32">
                <p>{T.loadingSubtitlesLabel || T.loadingLabel || "Loading subtitles..."}</p>
              </div>
            )}
            {error && !isLoading && (
              <div className="flex flex-col justify-center items-center h-32 text-red-500">
                <p>{T.errorLabel || "Error"}: {error}</p>
                <Button variant="link" onClick={handleFetchSubtitles} className="mt-2">
                  {T.retryLabel || "Try Again"}
                </Button>
              </div>
            )}
            {!isLoading && !error && subtitles.length === 0 && (
              <div className="flex justify-center items-center h-32 text-muted-foreground">
                <p>{T.noSubtitlesFoundLabel || "No subtitles. Select language & click Translate."}</p>
              </div>
            )}
            {!isLoading && !error && subtitles.length > 0 && subtitles.map((subtitle) => (
              <motion.div 
                key={subtitle.id}
                className={cn(
                  "p-2 sm:p-3 rounded-xl",
                  theme === "dark" ? "bg-zinc-800" : "bg-gray-100"
                )}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex flex-wrap sm:flex-nowrap items-center gap-2 sm:gap-0 sm:justify-between mb-3">
                  <div className="flex flex-wrap sm:flex-nowrap items-center gap-2 sm:gap-4">
                    <div className="flex items-center space-x-1">
                      <IonIcon icon={time} className="h-4 w-4 text-blue-500" />
                      <span className="text-xs text-muted-foreground">
                        {subtitle.startTime} - {subtitle.endTime} 
                      </span>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-2 text-xs text-blue-600 hover:text-blue-700 hover:bg-blue-50 dark:text-blue-400 dark:hover:bg-blue-900/20"
                      onClick={() => jumpToTime(subtitle.startTime)}
                    >
                      <IonIcon icon={caretForward} className="h-3 w-3 mr-1" />
                      {T.jumpToLabel}
                    </Button>
                  </div>
                </div>
                
                <div className="mb-3">
                  <label className="text-xs font-medium text-muted-foreground mb-1 block">
                    {T.originalSubtitleLabel}
                  </label>
                  <div className={cn(
                    "p-2 sm:p-3 rounded-lg text-sm",
                    theme === "dark" ? "bg-zinc-900" : "bg-gray-50"
                  )}>
                    {subtitle.text}
                  </div>
                </div>
                
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs font-medium text-muted-foreground">
                      {T.translatedSubtitleLabel}
                    </label>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-6 px-2 text-xs"
                      onClick={() => toggleEditMode(subtitle.id)}
                    >
                      <IonIcon icon={editingSubtitleId === subtitle.id ? checkmark : pencil} className="h-3 w-3 mr-1" />
                      {editingSubtitleId === subtitle.id ? T.saveLabel : T.editLabel}
                    </Button>
                  </div>
                  
                  {editingSubtitleId === subtitle.id ? (
                    <Textarea
                      value={subtitle.translation}
                      onChange={(e) => updateSubtitleTranslation(subtitle.id, e.target.value)}
                      className={cn(
                        "min-h-[60px] text-sm",
                        theme === "dark" ? "bg-zinc-800 border-zinc-700" : "bg-white"
                      )}
                    />
                  ) : (
                    <div className={cn(
                      "p-2 sm:p-3 rounded-lg text-sm",
                      theme === "dark" ? "bg-zinc-900 text-blue-400" : "bg-blue-50 text-blue-700"
                    )}>
                      {subtitle.translation}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
} 