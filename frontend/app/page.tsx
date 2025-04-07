// File: app/page.tsx
// Purpose: Main page component handling both Landing and App views with transitions.

'use client'

import { useState, useRef, useEffect, Suspense } from 'react'
import dynamic from 'next/dynamic'
import type { ParticleBackgroundRef, ParticleState } from '@/components/particle-background'
import VideoPlayer from "@/components/video-player" // Ensure VideoPlayer doesn't cause issues itself

// 直接导入粒子背景组件，不使用dynamic来避免ref传递问题
import ParticleBackground from '@/components/particle-background'

// 只动态导入其他不需要ref的组件
const LandingNav = dynamic(
  () => import('@/components/landing/LandingNav').then(mod => mod.LandingNav),
  { ssr: false }
)

const LandingHero = dynamic(
  () => import('@/components/landing/LandingHero').then(mod => mod.LandingHero),
  { ssr: false }
)

export default function HomePage() {
  // State to control which view is displayed: 'landing' or 'app'
  const [pageMode, setPageMode] = useState<'landing' | 'app'>('landing')

  // 简化动画状态管理，使用单一的状态
  const [particleState, setParticleState] = useState<ParticleState>('initial')
  
  // 添加一个状态锁，防止状态循环
  const [isStateTransitioning, setIsStateTransitioning] = useState(false)

  // State to control visibility and animation of landing/app content
  const [showLandingContent, setShowLandingContent] = useState(true) // For LandingNav/LandingHero opacity
  const [showUploadButton, setShowUploadButton] = useState(false) // For App upload button visibility/animation

  // App-specific state
  const [showPlayer, setShowPlayer] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  // Refs
  const particleRef = useRef<ParticleBackgroundRef>(null) // Ref for direct interaction
  const fileInputRef = useRef<HTMLInputElement>(null)
  const stateChangeTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // 清除任何可能存在的超时
  useEffect(() => {
    return () => {
      if (stateChangeTimeoutRef.current) {
        clearTimeout(stateChangeTimeoutRef.current);
      }
    };
  }, []);

  // --- Transition Handlers ---

  /**
   * Called when "Free Trial" button is clicked.
   * Initiates the transition from Landing to App view.
   */
  const handleTryFreeClick = () => {
    if (isStateTransitioning) {
      console.log('忽略重复点击，当前已在转场中');
      return; // 防止重复触发
    }
    
    console.log('开始转场动画，从登录页到应用页');
    // 首先设置状态锁，防止重复触发
    setIsStateTransitioning(true);
    // 开始隐藏登录页内容
    setShowLandingContent(false);
    // 延迟一帧后开始粒子动画，确保UI更新顺序正确
    requestAnimationFrame(() => {
      setParticleState('transitioningForward');
    });
  }

  /**
   * Callback executed when the particle background's animation state changes.
   */
  const handleAnimationStateChange = (newState: ParticleState) => {
    console.log('动画状态变化:', newState, '当前页面模式:', pageMode, '状态锁:', isStateTransitioning);
    
    // 记录当前状态以便能正确响应状态变化
    setParticleState(newState);
    
    if (newState === 'targetReached' && pageMode === 'landing') {
      console.log('前向转场完成，切换到应用模式');
      // 前向转场完成后，改变页面模式
      setPageMode('app');
      
      // 延迟显示上传按钮和释放状态锁，确保先完成页面模式切换
      setTimeout(() => {
        console.log('显示上传按钮');
        setShowUploadButton(true);
        setIsStateTransitioning(false); // 释放状态锁
      }, 200);
    } 
    else if (newState === 'initial' && pageMode === 'app') {
      console.log('重置动画完成，切换回登录页');
      // 重置动画完成，设置回登录页状态
      setPageMode('landing');
      setShowLandingContent(true);
      setShowPlayer(false);
      setSelectedFile(null);
      setShowUploadButton(false);
      setIsStateTransitioning(false); // 释放状态锁
    }
  }

  // --- App Interaction Handlers ---

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0]

      if (file.type.startsWith('video/')) {
        setSelectedFile(file)
        setShowUploadButton(false) // Hide upload button
        setShowPlayer(true) // Show video player component
      } else {
        alert("请选择一个有效的视频文件。") // Please select a valid video file.
      }

      // Clear input value to allow re-selecting the same file
      e.target.value = ''
    }
  }

  /**
   * Handles clicks on the main background area.
   * In 'app' mode, triggers the reset transition back to the landing page.
   */
  const handleBackgroundClick = (e: React.MouseEvent) => {
    // Only trigger reset if in 'app' mode and click is not on interactive elements
    if (pageMode === 'app' && !isStateTransitioning) {
      const target = e.target as HTMLElement
      // Check if the click target is the upload button or the video player container
      if (
        target.closest('#upload-button') ||
        target.closest('.video-player-container') || // Use a class specific to the player wrapper
        target.closest('canvas#particle-canvas') !== e.currentTarget // Ensure click is not on canvas itself if desired
      ) {
         // Click was on the button or player, do nothing for reset
         return;
      }

      // Clicked on the background area in app mode, initiate reset
      setIsStateTransitioning(true); // 设置状态锁，防止重复触发
      setShowUploadButton(false); // Immediately hide app elements or start their fade-out
      setShowPlayer(false);
      setParticleState('transitioningReset'); // Trigger the particle reset animation state
    }
  }

  // --- Rendering Logic ---

  return (
    // The main container div handles background clicks for reset trigger
    <div
      id="page-container"
      className="min-h-screen bg-black text-white overflow-hidden relative" // Added relative positioning
      onClick={handleBackgroundClick} // Attach click handler here
    >
      {/* 使用粒子背景组件 */}
      <Suspense fallback={<div className="fixed inset-0 bg-black" />}>
        <ParticleBackground
          animationState={particleState}
          onAnimationStateChange={handleAnimationStateChange}
          ref={particleRef}
        />
      </Suspense>

      {/* Landing Page Content - Conditionally rendered and animated */}
      <div
        id="landing-page-content"
        className={`absolute inset-0 min-h-screen flex flex-col z-10
          transition-opacity duration-500 ease-out
          ${showLandingContent ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
      >
        {/* 只在landing模式或视图切换过程中渲染内容 */}
        {(pageMode === 'landing' || particleState === 'transitioningForward') && (
          <>
            <LandingNav onTryFree={handleTryFreeClick} />
            <LandingHero onTryFree={handleTryFreeClick} />
          </>
        )}
      </div>

      {/* App Page Content - Conditionally rendered when pageMode is 'app' */}
      {/* 使用fade-in/fade-out动画确保平滑过渡 */}
      <div className={`transition-opacity duration-300 ${pageMode === 'app' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={handleFileChange}
        />

        {/* Upload Button - Animated */}
        {showUploadButton && (
          <div className="fixed inset-0 flex justify-center items-center z-10 animate-fade-in">
            <button
              id="upload-button"
              className="bg-white text-black font-semibold px-6 py-3 rounded-md flex items-center space-x-1.5 hover:bg-gray-200 transition-all duration-300 ease-in animate-pulse"
              onClick={handleUploadClick}
            >
              <span>上传视频</span> {/* Upload Video */}
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" x2="12" y1="3" y2="15"/>
              </svg>
            </button>
          </div>
        )}

        {/* Video Player */}
        {showPlayer && selectedFile && ( // Ensure selectedFile is also checked
          <div className="video-player-container fixed inset-0 z-20 flex justify-center items-center">
            {/* Added backdrop or background to prevent clicks passing through */}
            <div className="absolute inset-0 bg-black/50 backdrop-blur-sm"></div>
            <div className="w-full max-w-5xl p-4 z-30"> {/* Ensure player content is above backdrop */}
              <VideoPlayer initialFile={selectedFile} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}