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
  const [ignoreBackgroundReset, setIgnoreBackgroundReset] = useState<boolean>(false)

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
    
    // 重置所有与应用相关的状态
    setShowUploadButton(false);
    setShowPlayer(false);
    setSelectedFile(null);
    
    // 延迟设置粒子状态，确保UI更新顺序正确
    setTimeout(() => {
      setParticleState('transitioningForward');
    }, 50);
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
      
      // 延迟显示登录页内容，确保状态正确
      setTimeout(() => {
        setShowLandingContent(true);
        setShowPlayer(false);
        setSelectedFile(null);
        setShowUploadButton(false);
        setIsStateTransitioning(false); // 释放状态锁
      }, 200);
    }
  }

  // --- App Interaction Handlers ---

  const handleUploadClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    // 确保阻止事件冒泡
    e.stopPropagation();
    
    // 短暂设置忽略背景点击标志，防止文件选择器打开时触发背景点击
    setIgnoreBackgroundReset(true);
    
    // 打开文件选择器
    fileInputRef.current?.click();
    
    // 短暂延迟后重置忽略标志
    setTimeout(() => {
      setIgnoreBackgroundReset(false);
    }, 100);
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // 阻止事件冒泡
    e.stopPropagation();
    
    // 标记忽略背景点击一小段时间，以防文件选择后立即触发背景点击
    setIgnoreBackgroundReset(true);
    
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
    
    // 确保短暂延迟后重置忽略标志
    setTimeout(() => {
      setIgnoreBackgroundReset(false);
    }, 300);
  }

  /**
   * Handles clicks on the main background area.
   * In 'app' mode, triggers the reset transition back to the landing page.
   */
  const handleBackgroundClick = (e: React.MouseEvent) => {
    // 如果设置了忽略背景重置标志，重置它并立即返回
    if (ignoreBackgroundReset) {
      console.log('忽略背景点击，由上传按钮触发');
      setIgnoreBackgroundReset(false);
      return;
    }

    // 仅检查事件目标是否是上传按钮
    const target = e.target as HTMLElement;
    
    // Only trigger reset if in 'app' mode and click is not on interactive elements
    if (pageMode === 'app' && !isStateTransitioning) {
      // 简化检查，只查找上传按钮和视频播放器
      const isOnInteractiveElement = 
        target.closest('#upload-button') !== null ||
        target.closest('.video-player-container') !== null;
      
      // 如果点击不是在交互元素上，则触发重置
      if (!isOnInteractiveElement) {
        console.log('点击空白区域，开始逆向转场');
        setIsStateTransitioning(true); // 设置状态锁，防止重复触发
        
        // 立即隐藏上传按钮和视频播放器
        setShowUploadButton(false);
        setShowPlayer(false);
        
        // 确保UI更新后再设置粒子状态
        setTimeout(() => {
          // 触发粒子重置动画
          setParticleState('transitioningReset');
        }, 50);
      }
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
          onClick={(e) => {
            // 阻止文件输入点击事件冒泡
            e.stopPropagation();
          }}
        />

        {/* Upload Button - Animated */}
        {showUploadButton && (
          <div 
            className="fixed inset-0 flex justify-center items-center z-10 animate-fade-in"
            onClick={(e) => {
              // 只阻止事件冒泡，当点击发生在按钮上时
              if (e.target !== e.currentTarget) {
                e.stopPropagation();
              }
            }}
          >
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