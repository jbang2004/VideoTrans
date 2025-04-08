import React, { useState, useRef, ChangeEvent, useEffect, useCallback } from 'react';
// 假设 shadcn/ui 组件位于 @/components/ui
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { AspectRatio } from '@/components/ui/aspect-ratio';
import { Input } from '@/components/ui/input';
import { Slider } from "@/components/ui/slider";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

// 安装 lucide-react: npm install lucide-react
import { Upload, X, Pencil, Play, Pause, Volume2, Volume1, VolumeX, ArrowUp, Maximize, Minimize, Languages, Captions } from 'lucide-react';

// Helper function to format time (MM:SS) - unchanged
const formatTime = (timeInSeconds: number): string => {
    if (isNaN(timeInSeconds) || timeInSeconds === Infinity) { return '0:00'; }
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
};

// 主组件
export default function App() {
  // File and Preview State
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);

  // Player State
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isVolumePopoverOpen, setIsVolumePopoverOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Language and Subtitles State
  const [selectedLanguage, setSelectedLanguage] = useState('中文');
  const [subtitlesEnabled, setSubtitlesEnabled] = useState(false);
  const [isLanguagePopoverOpen, setIsLanguagePopoverOpen] = useState(false);

  // Available languages (示例)
  const availableLanguages = ['中文', 'English', '日本語', '한국어'];

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // --- Event Handlers ---
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    // Reset player and popover states
    setIsPlaying(false); setCurrentTime(0); setDuration(0); setIsVolumePopoverOpen(false);
    setIsLanguagePopoverOpen(false);
    // Reset subtitles status on new file selection
    setSubtitlesEnabled(false);
    // Note: Language selection is kept intentionally

    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      if (videoPreviewUrl) { URL.revokeObjectURL(videoPreviewUrl); }
      const previewUrl = URL.createObjectURL(file);
      setVideoPreviewUrl(previewUrl);
    } else {
      console.error("Please select a video file.");
      if (videoPreviewUrl) { URL.revokeObjectURL(videoPreviewUrl); }
      setVideoFile(null); setVideoPreviewUrl(null);
      if (fileInputRef.current) { fileInputRef.current.value = ''; }
    }
  };

  const handleDeleteVideo = useCallback(() => {
    // Reset player, popover, and subtitle states
    setIsPlaying(false); setCurrentTime(0); setDuration(0); setIsVolumePopoverOpen(false);
    setIsLanguagePopoverOpen(false);
    setSubtitlesEnabled(false); // Added subtitle reset for consistency
    // Note: Language selection is kept intentionally
    if (videoPreviewUrl) { URL.revokeObjectURL(videoPreviewUrl); }
    setVideoFile(null); setVideoPreviewUrl(null);
    if (fileInputRef.current) { fileInputRef.current.value = ''; }
  }, [videoPreviewUrl]);

  const triggerFileSelect = () => { fileInputRef.current?.click(); };

  const handleUpload = () => {
    if (!videoFile) { alert("请先选择一个视频文件！"); return; }
    console.log("Uploading:", videoFile.name);
    alert(`模拟上传: ${videoFile.name}`);
    handleDeleteVideo(); // Reset after upload simulation
  };


  // --- Player Controls ---
   const togglePlayPause = () => {
    if (!videoRef.current) return;
    if (isPlaying) { videoRef.current.pause(); } else { videoRef.current.play(); }
  };

  const handleSeek = (value: number[]) => {
    if (!videoRef.current) return;
    const newTime = value[0];
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const handleVolumeChange = (value: number[]) => {
    if (!videoRef.current) return;
    const newVolume = value[0];
    videoRef.current.volume = newVolume;
    videoRef.current.muted = newVolume === 0;
  };

  // --- Language and Subtitles Controls ---
  const handleLanguageSelect = (language: string) => {
    setSelectedLanguage(language);
    setIsLanguagePopoverOpen(false);
    console.log("Language selected:", language);
  };

  const toggleSubtitles = () => {
    setSubtitlesEnabled(prev => !prev);
    console.log("Subtitles toggled:", !subtitlesEnabled);
  };

  // --- Fullscreen Control ---
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(err => {
        console.error(`Error attempting fullscreen: ${err.message}`);
        alert(`无法进入全屏模式: ${err.message}`);
      });
    } else {
      if (document.exitFullscreen) { document.exitFullscreen(); }
    }
  };

  // --- Effect Listeners ---
  useEffect(() => {
    const handleFullscreenChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const updateTime = () => setCurrentTime(video.currentTime);
    const updateDuration = () => setDuration(video.duration);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);
    const updateVolumeState = () => { setVolume(video.volume); setIsMuted(video.muted); };
    video.addEventListener('timeupdate', updateTime);
    video.addEventListener('loadedmetadata', updateDuration);
    video.addEventListener('durationchange', updateDuration);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handleEnded);
    video.addEventListener('volumechange', updateVolumeState);
    updateVolumeState();
    return () => {
      video.removeEventListener('timeupdate', updateTime);
      video.removeEventListener('loadedmetadata', updateDuration);
      video.removeEventListener('durationchange', updateDuration);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('volumechange', updateVolumeState);
    };
  }, [videoPreviewUrl]);


 // Determine which volume icon to display
 const VolumeIcon = isMuted || volume === 0 ? VolumeX : volume < 0.5 ? Volume1 : Volume2;
 // Icon stroke width for main controls
 const iconStrokeWidth = 1.75;
 // Slider thumb size class
 const sliderThumbClass = "[&_[role=slider]]:w-2.5 [&_[role=slider]]:h-2.5";
 // Base slider track style
 const sliderTrackBase = "[&>span:first-child]:rounded-full [&>span:first-child]:bg-gray-600/50";
 // Base slider range (fill) style
 const sliderRangeBase = "[&>span:first-child>span]:bg-white [&>span:first-child>span]:group-hover:bg-gray-100 [&>span:first-child>span]:rounded-full";
 // Base slider thumb style
 const sliderThumbBase = "[&_[role=slider]]:bg-white [&_[role=slider]]:border-0 [&_[role=slider]]:shadow [&_[role=slider]]:focus-visible:ring-0 [&_[role=slider]]:focus-visible:ring-offset-0";
 // Base button style for controls
 const controlButtonBaseClass = "text-gray-200 bg-white/10 hover:bg-white/20 border-none";
 // Slider class constants
 const horizontalSliderClasses = `relative flex items-center select-none touch-none w-full h-4 group ${sliderThumbClass} ${sliderThumbBase} ${sliderTrackBase} [&>span:first-child]:h-1 ${sliderRangeBase} [&>span:first-child>span]:absolute [&>span:first-child>span]:h-full`;
 const verticalSliderClasses = `relative flex flex-col items-center select-none touch-none h-full w-4 group ${sliderThumbClass} ${sliderThumbBase} ${sliderTrackBase} [&>span:first-child]:w-1 ${sliderRangeBase} [&>span:first-child>span]:absolute [&>span:first-child>span]:w-full`;


  return (
    <TooltipProvider delayDuration={200}>
      {/* Outermost div: Solid background color */}
      <div
        ref={containerRef}
        className={`flex items-center justify-center min-h-screen p-4 font-sans bg-gray-900`}
      >
        {/* Card: No border, frosted glass background */}
        <Card className={`w-full max-w-2xl mx-auto bg-black/60 backdrop-blur-xl rounded-3xl text-gray-200 shadow-2xl transition-colors ${isFullscreen ? 'bg-black' : ''}`}>
          <CardContent className="p-4 md:p-6">
            <div className="mb-4">
              {/* AspectRatio: No border */}
              <AspectRatio ratio={16 / 9} className="rounded-xl overflow-hidden">
                {videoPreviewUrl ? (
                  <div className="relative w-full h-full group bg-black">
                    <video
                      ref={videoRef}
                      src={videoPreviewUrl}
                      className="object-contain w-full h-full block"
                      controls={false}
                      onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                      onClick={togglePlayPause}
                      muted={isMuted}
                    />
                    {/* Edit/Delete Buttons */}
                    <div className="absolute top-2.5 right-2.5 flex space-x-1.5 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        <Tooltip>
                            <TooltipTrigger asChild>
                                {/* Using fixed strokeWidth */}
                                <Button variant="secondary" size="icon" className="h-7 w-7 rounded-full bg-black/60 text-white/80 hover:text-white hover:bg-black/75 border-none">
                                    <Pencil className="h-3.5 w-3.5" strokeWidth={2} />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="bg-black/70 text-white border-none text-xs">
                                <p>编辑</p>
                            </TooltipContent>
                        </Tooltip>
                         <Tooltip>
                            <TooltipTrigger asChild>
                                {/* Using fixed strokeWidth */}
                                <Button variant="secondary" size="icon" className="h-7 w-7 rounded-full bg-black/60 text-white/80 hover:text-white hover:bg-black/75 border-none" onClick={handleDeleteVideo}>
                                    <X className="h-4 w-4" strokeWidth={2} />
                                </Button>
                            </TooltipTrigger>
                             <TooltipContent side="bottom" className="bg-black/70 text-white border-none text-xs">
                                <p>删除</p>
                            </TooltipContent>
                        </Tooltip>
                    </div>
                     {/* Centered play button overlay */}
                    {!isPlaying && (
                       <div className="absolute inset-0 flex items-center justify-center bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 cursor-pointer" onClick={togglePlayPause}>
                           <div className="bg-black/50 backdrop-blur-sm rounded-full p-3">
                             <Play className="h-10 w-10 text-white/90" fill="currentColor" strokeWidth={iconStrokeWidth} />
                           </div>
                       </div>
                    )}
                  </div>
                ) : (
                  // File selection trigger
                  <div
                    className="flex flex-col items-center justify-center w-full h-full border-2 border-dashed border-gray-600/40 rounded-xl cursor-pointer hover:bg-gray-700/40 transition-colors bg-black/30"
                    onClick={triggerFileSelect}
                    role="button" tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') triggerFileSelect(); }}
                  >
                    {/* Using fixed strokeWidth */}
                    <Upload className="h-12 w-12 text-gray-500/80 mb-2" strokeWidth={1.5} />
                    <p className="text-sm text-gray-400">点击或拖拽视频到此处</p>
                    <p className="text-xs text-gray-500">支持 MP4, AVI, MOV 等格式</p>
                  </div>
                )}
              </AspectRatio>
            </div>
            <Input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileChange} className="hidden" aria-hidden="true" />
          </CardContent>

          {/* Footer: Inherits Card background, no border-t */}
          <CardFooter className="flex items-center justify-between gap-x-2 md:gap-x-3 gap-y-2 p-3 md:px-4 md:py-3 rounded-b-3xl">
            {videoPreviewUrl ? (
              <>
                {/* Play/Pause Button */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="secondary" size="icon" className={`${controlButtonBaseClass} rounded-full h-8 w-8`} onClick={togglePlayPause}>
                      {isPlaying ? <Pause className="h-4 w-4" fill="currentColor"/> : <Play className="h-4 w-4" fill="currentColor"/>}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top"><p>{isPlaying ? "暂停" : "播放"}</p></TooltipContent>
                </Tooltip>

                {/* Time Display */}
                <span className="text-xs text-gray-300 font-mono tabular-nums mx-1 hidden sm:inline">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>

                {/* Progress Bar */}
                <div className="flex-grow mx-1">
                  {/* Using constant for classes */}
                  <Slider
                    value={[currentTime]} max={duration || 1} step={0.1}
                    onValueChange={handleSeek} aria-label="Video progress"
                    className={horizontalSliderClasses}
                  />
                </div>

                {/* Volume Control */}
                <Popover open={isVolumePopoverOpen} onOpenChange={setIsVolumePopoverOpen}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <PopoverTrigger asChild>
                        <Button variant="secondary" size="icon" className={`${controlButtonBaseClass} rounded-full h-8 w-8`} aria-label="音量">
                          <VolumeIcon className="h-4.5 w-4.5" strokeWidth={iconStrokeWidth} />
                        </Button>
                      </PopoverTrigger>
                    </TooltipTrigger>
                    <TooltipContent side="top"><p>音量</p></TooltipContent>
                  </Tooltip>
                  <PopoverContent side="top" align="center" sideOffset={10} className="h-28 w-auto p-2.5 bg-gray-900/80 backdrop-blur-md border border-gray-700/50 rounded-lg shadow-xl flex justify-center">
                     {/* Using constant for classes */}
                    <Slider
                      orientation="vertical"
                      value={[isMuted ? 0 : volume]} max={1} step={0.05}
                      onValueChange={handleVolumeChange} aria-label="音量调节"
                      className={verticalSliderClasses}
                    />
                  </PopoverContent>
                </Popover>

                {/* Language Selection */}
                <Popover open={isLanguagePopoverOpen} onOpenChange={setIsLanguagePopoverOpen}>
                   <Tooltip>
                     <TooltipTrigger asChild>
                       <PopoverTrigger asChild>
                         <Button variant="secondary" className={`${controlButtonBaseClass} rounded-full h-8 px-3 text-xs flex items-center gap-1.5`}>
                            <Languages className="h-4 w-4" strokeWidth={iconStrokeWidth} />
                            <span>{selectedLanguage}</span>
                         </Button>
                       </PopoverTrigger>
                     </TooltipTrigger>
                     <TooltipContent side="top"><p>选择语言</p></TooltipContent>
                   </Tooltip>
                   <PopoverContent side="top" align="center" sideOffset={8} className="w-auto p-1 bg-gray-900/80 backdrop-blur-md border border-gray-700/50 rounded-lg shadow-xl">
                     {availableLanguages.map((lang) => (
                       <Button key={lang} variant="ghost" className={`w-full justify-start h-8 px-2 text-xs ${selectedLanguage === lang ? 'bg-gray-700/70 text-white' : 'text-gray-300 hover:bg-gray-700/50 hover:text-white'}`} onClick={() => handleLanguageSelect(lang)}>
                         {lang}
                       </Button>
                     ))}
                   </PopoverContent>
                 </Popover>

                 {/* Subtitles Toggle */}
                 <Tooltip>
                   <TooltipTrigger asChild>
                     <Button variant="secondary" className={`${controlButtonBaseClass} rounded-full h-8 px-3 text-xs flex items-center gap-1.5`} onClick={toggleSubtitles}>
                       <Captions className="h-4 w-4" strokeWidth={iconStrokeWidth} />
                       <span>{subtitlesEnabled ? '开' : '关'}</span>
                     </Button>
                   </TooltipTrigger>
                   <TooltipContent side="top"><p>字幕</p></TooltipContent>
                 </Tooltip>

                {/* Fullscreen Button */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="secondary" size="icon" className={`${controlButtonBaseClass} rounded-full h-8 w-8`} onClick={toggleFullscreen}>
                      {isFullscreen ? <Minimize className="h-4.5 w-4.5" strokeWidth={iconStrokeWidth} /> : <Maximize className="h-4.5 w-4.5" strokeWidth={iconStrokeWidth} />}
                    </Button>
                  </TooltipTrigger>
                   <TooltipContent side="top"><p>{isFullscreen ? "退出全屏" : "全屏"}</p></TooltipContent>
                </Tooltip>

                {/* Upload Button */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button size="icon" className="rounded-full bg-blue-600 text-white hover:bg-blue-500 h-8 w-8 shadow-md border-none ml-1 md:ml-2" onClick={handleUpload} disabled={!videoFile}>
                      <ArrowUp className="h-4.5 w-4.5" strokeWidth={iconStrokeWidth} />
                    </Button>
                  </TooltipTrigger>
                   <TooltipContent side="top" align="end"><p>{videoFile ? "上传视频" : "请先选择视频"}</p></TooltipContent>
                </Tooltip>
              </>
            ) : (
              // Placeholder when no video selected
              <>
                <span className="text-gray-500 text-xs flex-grow">选择视频后将显示播放器控件</span>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button size="icon" className="rounded-full bg-white/10 text-gray-200 hover:bg-white/20 h-8 w-8 border-none" onClick={triggerFileSelect}>
                      <Upload className="h-4.5 w-4.5" strokeWidth={iconStrokeWidth} />
                    </Button>
                  </TooltipTrigger>
                   <TooltipContent side="top" align="end"><p>选择视频</p></TooltipContent>
                </Tooltip>
              </>
            )}
          </CardFooter>
        </Card>
      </div>
    </TooltipProvider>
  );
}
