import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useLanguage } from "@/hooks/use-language";
import { useTheme } from "@/hooks/use-theme";
import { useAuth } from "@/hooks/use-auth";
import { useMediaQuery } from "@/hooks/use-media-query";
import { useVideoUpload } from "@/hooks/use-video-upload";
import { useSubtitles } from "@/hooks/use-subtitles";
import { useVideoPlayer } from "@/hooks/use-video-player";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabaseClient";
import { getTranslations, Language as AppLanguage } from "@/lib/translations";

// Import the new components
import VideoPanel from "@/components/video-translation/VideoPanel";
import SubtitlesPanel from "@/components/video-translation/SubtitlePanel";
import { BlurFade } from "@/components/magicui/blur-fade";

const FIXED_TASK_ID = "29a49af4-600f-4613-a1b4-31a4d6e1c210";

export default function VideoTranslation() {
  const { language: currentInterfaceLanguage } = useLanguage();
  const { theme } = useTheme();
  const T = getTranslations(currentInterfaceLanguage as AppLanguage);

  const { user: currentUser, session: currentSession, loading: authLoading } = useAuth();
  const isMobile = useMediaQuery('(max-width: 767px)');
  
  const [targetLanguage, setTargetLanguage] = useState<string>("en");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [displaySubtitlesPanel, setDisplaySubtitlesPanel] = useState<boolean>(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const subtitlesContainerRef = useRef<HTMLDivElement>(null);

  const {
    isUploading,
    uploadProgress,
    uploadComplete,
    videoPreviewUrl,
    uploadError,
    initiateUpload,
    resetUploadState: resetVideoUploadHookState
  } = useVideoUpload();

  const {
    isPlaying,
    setIsPlaying,
    togglePlayback,
    jumpToTime,
  } = useVideoPlayer({ videoRef });

  const {
    subtitles,
    showSubtitles, 
    editingSubtitleId,
    isLoadingSubtitles,
    subtitleError,   
    fetchSubtitles,  
    updateSubtitleTranslation,
    toggleEditMode,
    closeSubtitlesPanel: closeSubtitlesPanelHook,
    resetSubtitlesState,
  } = useSubtitles({ 
    loadCondition: uploadComplete && !!videoPreviewUrl && displaySubtitlesPanel,
  });

  useEffect(() => {
    if (uploadError) {
      alert(T.alertMessages.uploadFailed(uploadError.message));
    }
  }, [uploadError, T.alertMessages]);

  useEffect(() => {
    if (videoRef.current && videoPreviewUrl) {
      videoRef.current.src = videoPreviewUrl;
    }
  }, [videoPreviewUrl]);
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      resetVideoUploadHookState();
      resetSubtitlesState();
      if (videoRef.current) {
        videoRef.current.src = "";
      }
      setIsPlaying(false);
      setDisplaySubtitlesPanel(false); 
      if (showSubtitles) closeSubtitlesPanelHook(); 
      startActualUploadProcess(file);
    }
  };

  // This function is passed to VideoPanel to trigger the hidden file input
  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };
  
  const startActualUploadProcess = async (fileToUpload: File) => {
    if (!fileToUpload) {
      alert(T.alertMessages.selectVideoFirst);
      return;
    }
    if (authLoading) {
      alert("Auth loading, please wait...");
      return;
    }
    let userIdForUpload = currentUser?.id;
    let sessionAccessToken = currentSession?.access_token;
    if (!userIdForUpload || !sessionAccessToken) {
      alert(T.alertMessages.userInfoIncomplete);
      const { data: { session: newSession }, error: sessionError } = await supabase.auth.getSession();
      const { data: { user: newUser }, error: userError } = await supabase.auth.getUser();
      if (sessionError || userError || !newSession || !newUser) {
        alert(T.alertMessages.userInfoRefreshFailed);
        return;
      }
      if (newUser?.id && newSession?.access_token) {
        userIdForUpload = newUser.id;
        sessionAccessToken = newSession.access_token;
      } else {
        alert(T.alertMessages.sessionTokenInvalid);
        return;
      }
    }
    await initiateUpload(fileToUpload, { id: userIdForUpload } as any, sessionAccessToken); 
  };

  const resetFullUploadAndPlayer = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.src = "";
    }
    setSelectedFile(null);
    resetVideoUploadHookState();
    resetSubtitlesState();
    setIsPlaying(false);
    setDisplaySubtitlesPanel(false); 
    if(fileInputRef.current) fileInputRef.current.value = "";
  };
  
  const handlePreprocessingTrigger = () => {
    if (uploadComplete) {
      setDisplaySubtitlesPanel(true); 
    } else {
      alert(T.alertMessages.uploadNotComplete);
    }
  };
  
  const startGenerating = () => {
    alert(T.alertMessages.generatingVideo);
  };

  const getLanguageLabel = (value: string) => {
    const option = T.languageOptions.find(lang => lang.value === value);
    return option?.label || "";
  };

  const handleCloseSubtitlesPanel = () => {
    closeSubtitlesPanelHook(); 
    setDisplaySubtitlesPanel(false); 
  };

  return (
    <motion.div
      className="container mx-auto px-2 sm:px-4 py-12 overflow-x-hidden"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
    >
      <div className={cn(
        "flex flex-wrap justify-center",
        isMobile ? "mx-auto" : "-mx-2 md:-mx-3",
        isMobile && displaySubtitlesPanel ? "items-center flex-col" : "items-start"
      )}>
        <BlurFade
          layout
          className={cn(
            "px-2 md:px-3 mb-8",
            isMobile ? "w-full mx-auto max-w-md" :
              (displaySubtitlesPanel ? "md:mx-0 w-full max-w-md md:shrink-0" : "w-full max-w-md mx-auto")
          )}
          delay={0.25}
          inView={true}
        >
          <VideoPanel
            theme={theme}
            selectedFile={selectedFile}
            videoPreviewUrl={videoPreviewUrl}
            isUploading={isUploading}
            uploadProgress={uploadProgress}
            uploadComplete={uploadComplete}
            isPlaying={isPlaying}
            displaySubtitlesPanel={displaySubtitlesPanel} 
            translations={T}
            handleUploadClick={triggerFileInput} 
            resetUpload={resetFullUploadAndPlayer}
            togglePlayback={togglePlayback} 
            handlePreprocessingTrigger={handlePreprocessingTrigger}
            startGenerating={startGenerating}
            videoRef={videoRef}
            fileInputRef={fileInputRef}
          />
        </BlurFade>
        
        <AnimatePresence mode="popLayout">
          {displaySubtitlesPanel && ( 
            <BlurFade
              layout
              className={cn(
                "w-full px-2 md:px-3 mx-auto",
                isMobile ? "mt-8 max-w-xl" : "mt-0 md:mx-0 md:basis-[36rem] md:grow md:shrink h-full"
              )}
              inView={true}
              delay={0.1}
              direction={isMobile ? "up" : "left"}
              offset={isMobile ? 20 : 50}
              duration={0.5}
            >
              <SubtitlesPanel
                theme={theme}
                subtitles={subtitles}
                editingSubtitleId={editingSubtitleId}
                targetLanguage={targetLanguage}
                translations={T}
                isMobile={isMobile}
                isLoading={isLoadingSubtitles} 
                error={subtitleError}        
                getLanguageLabel={getLanguageLabel}
                jumpToTime={jumpToTime}
                updateSubtitleTranslation={updateSubtitleTranslation}
                toggleEditMode={toggleEditMode}
                setTargetLanguage={setTargetLanguage}
                fetchSubtitles={fetchSubtitles}   
                closeSubtitlesPanel={handleCloseSubtitlesPanel} 
                subtitlesContainerRef={subtitlesContainerRef}
                currentTaskId={FIXED_TASK_ID} 
              />
            </BlurFade>
          )}
        </AnimatePresence>
      </div>
      
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleFileChange}
      />
    </motion.div>
  );
}