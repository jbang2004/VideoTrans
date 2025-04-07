'use client'

import React, { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'
import { createCircleTextureSoft } from '@/lib/utils'
import { useParticleAnimation, ParticleState } from './hooks/useParticleAnimation'

export type { ParticleState }

export interface ParticleBackgroundRef {
  startTransition: () => void
  resetTransition: () => void
  getAnimationState: () => ParticleState
}

interface ParticleBackgroundProps {
  onTransitionComplete?: () => void
  onReset?: () => void
  animationState?: ParticleState
  onAnimationStateChange?: (newState: ParticleState) => void
}

const ParticleBackgroundComponent = React.forwardRef<ParticleBackgroundRef, ParticleBackgroundProps>(({
  onTransitionComplete,
  onReset,
  animationState: externalAnimationState,
  onAnimationStateChange
}, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const requestRef = useRef<number>()
  const lastReportedStateRef = useRef<ParticleState | null>(null)
  
  // 使用useState以确保组件只在客户端渲染时执行
  const [isMounted, setIsMounted] = useState(false)
  
  // 确保代码只在客户端执行
  useEffect(() => {
    setIsMounted(true)
  }, [])
  
  // 使用自定义hook管理粒子动画
  const { scene, camera, updateParticles, animationState, setAnimationState } = useParticleAnimation({
    initialState: externalAnimationState || 'initial',
    onAnimationComplete: (state) => {
      // 添加日志，便于调试
      console.log('粒子动画完成状态:', state);
      
      // 设置完成状态并直接通知外部
      lastReportedStateRef.current = state;
      
      // 通知外部状态变化，这是最关键的步骤
      console.log('通知外部状态变化:', state);
      onAnimationStateChange?.(state);
      
      // 其他回调可以在状态通知后执行
      if (state === 'targetReached') {
        console.log('目标状态达成，触发完成回调');
        onTransitionComplete?.();
      } else if (state === 'initial') {
        console.log('初始状态达成，触发重置回调');
        onReset?.();
      }
    }
  })
  
  // 监听外部传入的animationState变化
  useEffect(() => {
    if (!isMounted) return;
    
    if (externalAnimationState && externalAnimationState !== animationState) {
      console.log('外部状态变化:', externalAnimationState, '内部状态:', animationState);
      
      // 避免循环状态变化
      if ((externalAnimationState === 'transitioningForward' && animationState === 'targetReached') ||
          (externalAnimationState === 'transitioningReset' && animationState === 'initial')) {
        console.log('避免状态循环，跳过状态设置');
        return;
      }
      
      console.log('设置新状态:', externalAnimationState);
      setAnimationState(externalAnimationState);
      lastReportedStateRef.current = externalAnimationState;
    }
  }, [externalAnimationState, animationState, setAnimationState, isMounted])
  
  // 导出状态和控制函数供外部使用
  const startTransition = () => {
    if (animationState === 'initial') {
      setAnimationState('transitioningForward')
    }
  }
  
  const resetTransition = () => {
    if (animationState === 'targetReached') {
      setAnimationState('transitioningReset')
    }
  }
  
  // 初始化Three.js
  useEffect(() => {
    if (!canvasRef.current || !isMounted) return
    
    // 创建渲染器
    rendererRef.current = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true // 保留绘制缓冲区，提高视觉连续性
    })
    
    const handleResize = () => {
      if (!rendererRef.current || !camera) return
      
      camera.aspect = window.innerWidth / window.innerHeight
      camera.updateProjectionMatrix()
      rendererRef.current.setSize(window.innerWidth, window.innerHeight)
    }
    
    rendererRef.current.setPixelRatio(window.devicePixelRatio)
    rendererRef.current.setSize(window.innerWidth, window.innerHeight)
    
    window.addEventListener('resize', handleResize)
    
    // 动画循环
    const animate = () => {
      requestRef.current = requestAnimationFrame(animate)
      
      if (scene && camera && rendererRef.current) {
        updateParticles()
        rendererRef.current.render(scene, camera)
      }
    }
    
    animate()
    
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current)
      }
      window.removeEventListener('resize', handleResize)
      rendererRef.current?.dispose()
    }
  }, [scene, camera, updateParticles, isMounted])
  
  // 暴露方法给父组件
  React.useImperativeHandle(
    ref,
    () => ({
      startTransition,
      resetTransition,
      getAnimationState: () => animationState
    }),
    [animationState]
  )
  
  // 如果还没有挂载，返回一个占位符
  if (!isMounted) {
    return <div className="fixed inset-0 bg-black" />
  }
  
  return <canvas ref={canvasRef} id="particle-canvas" className="fixed inset-0 z-0" />
})

ParticleBackgroundComponent.displayName = 'ParticleBackground'

// 导出组件
export default ParticleBackgroundComponent
// 命名导出以兼容动态导入
export const ParticleBackground = ParticleBackgroundComponent