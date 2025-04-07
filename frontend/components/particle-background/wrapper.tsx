'use client'

import { useEffect, useState, useRef } from 'react'
import ParticleBackgroundComponent, { ParticleState, ParticleBackgroundRef } from './index'

interface ParticleBackgroundWrapperProps {
  startTransition: boolean
  onTransitionComplete?: () => void
  onReset?: () => void
}

export default function ParticleBackgroundWrapper({
  startTransition,
  onTransitionComplete,
  onReset
}: ParticleBackgroundWrapperProps) {
  const [isInitialRender, setIsInitialRender] = useState(true)
  const [animationState, setAnimationState] = useState<ParticleState>('initial')
  const particleRef = useRef<ParticleBackgroundRef>(null)

  // 监听startTransition属性变化，触发转场动画
  useEffect(() => {
    if (isInitialRender) {
      setIsInitialRender(false)
      return
    }
    
    if (startTransition && animationState === 'initial') {
      // 开始粒子转场动画
      setAnimationState('transitioningForward')
    }
  }, [startTransition, animationState, isInitialRender])

  // 触发动画状态变化的处理函数
  const handleAnimationStateChange = (newState: ParticleState) => {
    setAnimationState(newState)
    
    if (newState === 'targetReached' && onTransitionComplete) {
      onTransitionComplete()
    } else if (newState === 'initial' && onReset) {
      onReset()
    }
  }

  return (
    <ParticleBackgroundComponent 
      ref={particleRef}
      animationState={animationState}
      onAnimationStateChange={handleAnimationStateChange}
      onTransitionComplete={onTransitionComplete}
      onReset={onReset}
    />
  )
} 