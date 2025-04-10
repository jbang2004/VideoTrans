'use client'

import { useRef, useEffect, useState, useCallback } from 'react'
import * as THREE from 'three'
import { createCircleTextureSoft } from '@/lib/utils'

export type ParticleState = 'initial' | 'transitioningForward' | 'targetReached' | 'transitioningReset'

export interface UseParticleAnimationProps {
  initialState?: ParticleState
  onAnimationComplete?: (state: ParticleState) => void
}

export function useParticleAnimation({
  initialState = 'initial',
  onAnimationComplete
}: UseParticleAnimationProps) {
  const [animationState, setAnimationState] = useState<ParticleState>(initialState)
  const isClient = typeof window !== 'undefined'
  
  // 包装设置状态函数，添加日志
  const setAnimationStateWithLog = (newState: ParticleState) => {
    console.log(`设置粒子动画状态：${animationState} -> ${newState}`);
    setAnimationState(newState);
  }
  
  // 创建相机，确保在SSR阶段使用默认比例
  const defaultAspect = 16 / 9 // 默认宽高比
  const cameraRef = useRef(new THREE.PerspectiveCamera(
    75, 
    isClient ? window.innerWidth / window.innerHeight : defaultAspect, 
    1, 
    10000
  ))
  
  const sceneRef = useRef(new THREE.Scene())
  const particlesRef = useRef<THREE.Points | null>(null)
  
  // 配置参数
  const configRef = useRef({
    particleCount: 4000,
    separation: 110,
    targetParticleScale: 1.0,
    transitionDuration: 1200, // 毫秒，稍微增加转场时间以确保平滑过渡
    resetDuration: 1200, // 毫秒，同样增加时间
    waveSpeed: 0.05, // 波浪速度，控制粒子流动
  })
  
  // 动画状态引用
  const prevAnimationStateRef = useRef<ParticleState>(initialState)
  const countRef = useRef(0)
  const stateChangeTimeRef = useRef(0)
  const animationCompletedRef = useRef<{[key in ParticleState]?: boolean}>({})
  const stateChangeLockedRef = useRef(false)
  
  // 网格维度
  const gridDimensionsRef = useRef({
    amountX: 0,
    amountZ: 0,
    numParticlesActual: 0,
    gridWidth: 0,
    gridDepth: 0,
  })
  
  // 插值状态引用
  const interpolationProgressRef = useRef(0)
  const transitionStartTimeRef = useRef(0)
  
  // 存储基础网格位置和插值位置
  const baseGridPositionsRef = useRef<Float32Array | null>(null)
  const currentInterpolationStartPosRef = useRef<Float32Array | null>(null)
  const currentInterpolationStartScaleRef = useRef<Float32Array | null>(null)
  const targetPositionsRef = useRef<Float32Array | null>(null)
  const targetRotationRef = useRef<THREE.Quaternion | null>(null)
  
  // 计算网格尺寸
  const calculateGridDimensions = useCallback(() => {
    if (!isClient) return; // 服务器端不执行
    
    const { particleCount, separation } = configRef.current
    const aspectRatio = (window.innerWidth > 0 && window.innerHeight > 0) 
      ? (window.innerWidth / window.innerHeight) 
      : 1
    
    // 计算网格尺寸
    const amountX = Math.floor(Math.sqrt(particleCount * aspectRatio))
    const safaAmountX = Math.max(1, amountX)
    const amountZ = Math.floor(particleCount / safaAmountX)
    const safeAmountZ = Math.max(1, amountZ)
    
    const numParticlesActual = safaAmountX * safeAmountZ
    const gridWidth = safaAmountX * separation
    const gridDepth = safeAmountZ * separation
    
    gridDimensionsRef.current = {
      amountX: safaAmountX,
      amountZ: safeAmountZ,
      numParticlesActual,
      gridWidth,
      gridDepth,
    }
  }, [isClient])
  
  // 定义目标旋转
  const defineTargetRotation = useCallback(() => {
    targetRotationRef.current = new THREE.Quaternion()
    const rotationAxisX = new THREE.Vector3(1, 0, 0)
    const targetTiltAngle = -25 * Math.PI / 180 // -25度
    targetRotationRef.current.setFromAxisAngle(rotationAxisX, targetTiltAngle)
  }, [])
  
  // 初始化场景
  useEffect(() => {
    if (!isClient) return; // 服务器端不执行
    
    // 设置相机
    const camera = cameraRef.current
    camera.position.y = 700
    camera.position.z = 1200
    camera.rotation.x = -0.3
    
    // 计算网格尺寸
    calculateGridDimensions()
    
    // 定义目标旋转
    defineTargetRotation()
    
    // 创建粒子系统
    const { numParticlesActual, gridWidth, gridDepth, amountX, amountZ } = gridDimensionsRef.current
    const { separation } = configRef.current
    
    // 创建粒子
    if (numParticlesActual <= 0) return
    
    const positions = new Float32Array(numParticlesActual * 3)
    const scales = new Float32Array(numParticlesActual)
    
    let i = 0, j = 0
    for (let ix = 0; ix < amountX; ix++) {
      for (let iz = 0; iz < amountZ; iz++) {
        if (j + 2 >= positions.length || i >= scales.length) break
        positions[j] = ix * separation - (gridWidth / 2)
        positions[j + 1] = 0
        positions[j + 2] = iz * separation - (gridDepth / 2)
        scales[i] = 1
        i++
        j += 3
      }
      if (j + 2 >= positions.length || i >= scales.length) break
    }
    
    // 保存基础网格位置
    baseGridPositionsRef.current = new Float32Array(positions)
    
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('scale', new THREE.BufferAttribute(scales, 1))
    
    // 将属性设置为动态使用模式
    const positionAttribute = geometry.getAttribute('position') as THREE.BufferAttribute
    const scaleAttribute = geometry.getAttribute('scale') as THREE.BufferAttribute
    positionAttribute.setUsage(THREE.DynamicDrawUsage)
    scaleAttribute.setUsage(THREE.DynamicDrawUsage)
    
    // 使用柔和的圆形纹理，确保正确设置纹理参数
    const texture = createCircleTextureSoft()
    
    // 创建材质
    const material = new THREE.PointsMaterial({
      size: 25,
      map: texture,
      sizeAttenuation: true,
      vertexColors: false,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.8,
      depthWrite: false,
      color: new THREE.Color(0x0047AB) // 更深的蓝色(钴蓝色)
    })
    
    // 创建粒子系统
    const particles = new THREE.Points(geometry, material)
    sceneRef.current.add(particles)
    particlesRef.current = particles
    
    // 清除函数
    return () => {
      geometry.dispose()
      material.dispose()
      texture.dispose()
      sceneRef.current.remove(particles)
    }
  }, [isClient, calculateGridDimensions, defineTargetRotation])
  
  // 准备目标位置（用于转场）
  useEffect(() => {
    if (!isClient) return;
    
    if (prevAnimationStateRef.current !== animationState) {
      console.log(`动画状态变化: ${prevAnimationStateRef.current} -> ${animationState}`);
      
      // 避免在过渡中再次触发状态变化，但允许从中间状态到目标状态的转换
      const allowStateChange = 
        !stateChangeLockedRef.current || 
        animationState === 'targetReached' || 
        animationState === 'initial';
        
      if (!allowStateChange) {
        console.log('状态锁定中，跳过处理');
        return;
      }
      
      prevAnimationStateRef.current = animationState;
      stateChangeTimeRef.current = Date.now();
      
      // 重置该状态的完成标志
      animationCompletedRef.current = {}; // 重置所有完成标志
      animationCompletedRef.current[animationState] = false;
      
      // 处理前向转场
      if (animationState === 'transitioningForward') {
        if (!particlesRef.current) {
          console.log('粒子未初始化，无法开始转场');
          return;
        }
        
        console.log('开始前向转场动画');
        stateChangeLockedRef.current = true;
        transitionStartTimeRef.current = Date.now();
        interpolationProgressRef.current = 0;
        
        if (!targetPositionsRef.current && baseGridPositionsRef.current) {
          const { numParticlesActual } = gridDimensionsRef.current;
          
          targetPositionsRef.current = new Float32Array(numParticlesActual * 3);
          const targetVector = new THREE.Vector3();
          
          for (let i = 0; i < numParticlesActual; i++) {
            const i3 = i * 3;
            
            if (i3 + 2 >= baseGridPositionsRef.current.length || 
                i3 + 2 >= targetPositionsRef.current.length) break;
                
            targetVector.set(
              baseGridPositionsRef.current[i3],
              baseGridPositionsRef.current[i3 + 1],
              baseGridPositionsRef.current[i3 + 2]
            );
            
            // 应用旋转
            if (targetRotationRef.current) {
              targetVector.applyQuaternion(targetRotationRef.current);
            }
            
            targetPositionsRef.current[i3] = targetVector.x;
            targetPositionsRef.current[i3 + 1] = targetVector.y;
            targetPositionsRef.current[i3 + 2] = targetVector.z;
          }
        }
        
        // 保存当前位置
        if (particlesRef.current && particlesRef.current.geometry) {
          const geometry = particlesRef.current.geometry;
          currentInterpolationStartPosRef.current = new Float32Array(geometry.attributes.position.array);
          currentInterpolationStartScaleRef.current = new Float32Array(geometry.attributes.scale.array);
        }
      }
      
      // 处理重置转场
      else if (animationState === 'transitioningReset') {
        if (!particlesRef.current) {
          console.log('粒子未初始化，无法开始重置');
          return;
        }
        
        console.log('开始重置动画');
        stateChangeLockedRef.current = true;
        transitionStartTimeRef.current = Date.now();
        interpolationProgressRef.current = 0;
        
        // 保存当前位置
        if (particlesRef.current && particlesRef.current.geometry) {
          const geometry = particlesRef.current.geometry;
          currentInterpolationStartPosRef.current = new Float32Array(geometry.attributes.position.array);
          currentInterpolationStartScaleRef.current = new Float32Array(geometry.attributes.scale.array);
        }
      } else {
        // 其他状态（target或initial）完成时，解锁状态变化
        console.log(`到达稳定状态: ${animationState}，解锁状态变化`);
        stateChangeLockedRef.current = false;
        // 确保清除所有动画标志
        animationCompletedRef.current = {};
      }
    }
  }, [animationState, isClient])
  
  // 更新粒子
  const updateParticles = useCallback(() => {
    if (!isClient || !particlesRef.current) return; // 服务器端不执行
    
    const particles = particlesRef.current;
    const geometry = particles.geometry;
    const positions = geometry.attributes.position.array as Float32Array;
    const scales = geometry.attributes.scale.array as Float32Array;
    let needsUpdate = false;
    
    const { amountX, amountZ } = gridDimensionsRef.current;
    const { waveSpeed, transitionDuration, resetDuration } = configRef.current;
    
    // 前向转场动画
    if (animationState === 'transitioningForward') {
      const elapsedTime = Date.now() - transitionStartTimeRef.current
      const progress = Math.min(1.0, elapsedTime / transitionDuration)
      
      // 使用平滑的ease-in-out曲线
      const easedProgress = progress < 0.5 ? 2 * Math.pow(progress, 2) : 1 - Math.pow(-2 * progress + 2, 2) / 2
      interpolationProgressRef.current = easedProgress
      
      // 插值到目标位置
      if (currentInterpolationStartPosRef.current && targetPositionsRef.current) {
        let i = 0, j = 0
        for (let ix = 0; ix < amountX; ix++) {
          for (let iz = 0; iz < amountZ; iz++) {
            if (j + 2 >= positions.length || j + 2 >= currentInterpolationStartPosRef.current.length || j + 2 >= targetPositionsRef.current.length) break
            
            positions[j] = currentInterpolationStartPosRef.current[j] + (targetPositionsRef.current[j] - currentInterpolationStartPosRef.current[j]) * easedProgress
            positions[j + 1] = currentInterpolationStartPosRef.current[j + 1] + (targetPositionsRef.current[j + 1] - currentInterpolationStartPosRef.current[j + 1]) * easedProgress
            positions[j + 2] = currentInterpolationStartPosRef.current[j + 2] + (targetPositionsRef.current[j + 2] - currentInterpolationStartPosRef.current[j + 2]) * easedProgress
            
            j += 3
          }
          if (j + 2 >= positions.length) break
        }
        
        needsUpdate = true
      }
      
      // 如果到达目标，切换状态
      if (progress >= 1.0 && !animationCompletedRef.current[animationState]) {
        console.log('前向动画完成，切换到目标状态');
        animationCompletedRef.current[animationState] = true;
        setAnimationStateWithLog('targetReached');
        
        // 确保下一帧立即触发回调，并解锁状态
        setTimeout(() => {
          stateChangeLockedRef.current = false;
          onAnimationComplete?.('targetReached');
        }, 50);
        
        needsUpdate = true;
      }
    }
    
    // 重置动画
    else if (animationState === 'transitioningReset') {
      const elapsedTime = Date.now() - transitionStartTimeRef.current
      const progress = Math.min(1.0, elapsedTime / resetDuration)
      
      // 使用平滑的ease-in-out曲线
      const easedProgress = progress < 0.5 ? 2 * Math.pow(progress, 2) : 1 - Math.pow(-2 * progress + 2, 2) / 2
      interpolationProgressRef.current = easedProgress
      
      // 插值回初始位置
      if (currentInterpolationStartPosRef.current && baseGridPositionsRef.current) {
        let i = 0, j = 0
        for (let ix = 0; ix < amountX; ix++) {
          for (let iz = 0; iz < amountZ; iz++) {
            if (j + 2 >= positions.length || j + 2 >= currentInterpolationStartPosRef.current.length || j + 2 >= baseGridPositionsRef.current.length) break
            
            positions[j] = currentInterpolationStartPosRef.current[j] + (baseGridPositionsRef.current[j] - currentInterpolationStartPosRef.current[j]) * easedProgress
            positions[j + 1] = currentInterpolationStartPosRef.current[j + 1] + (baseGridPositionsRef.current[j + 1] - currentInterpolationStartPosRef.current[j + 1]) * easedProgress
            positions[j + 2] = currentInterpolationStartPosRef.current[j + 2] + (baseGridPositionsRef.current[j + 2] - currentInterpolationStartPosRef.current[j + 2]) * easedProgress
            
            j += 3
          }
          if (j + 2 >= positions.length) break
        }
        
        needsUpdate = true
      }
      
      // 如果回到初始状态，则停止
      if (progress >= 1.0 && !animationCompletedRef.current[animationState]) {
        console.log('重置动画完成，切换到初始状态');
        animationCompletedRef.current[animationState] = true;
        setAnimationStateWithLog('initial');
        
        // 确保下一帧立即触发回调，并解锁状态
        setTimeout(() => {
          stateChangeLockedRef.current = false;
          onAnimationComplete?.('initial');
        }, 50);
        
        needsUpdate = true;
      }
    }
    
    // 初始状态下的粒子流动效果
    else if (animationState === 'initial') {
      countRef.current += waveSpeed;
      
      if (baseGridPositionsRef.current) {
        let i = 0, j = 0;
        
        for (let ix = 0; ix < amountX; ix++) {
          for (let iz = 0; iz < amountZ; iz++) {
            if (j + 2 >= positions.length || i >= scales.length || j + 2 >= baseGridPositionsRef.current.length) break;
            
            // 保持X和Z不变，只在Y轴上应用波浪效果
            positions[j] = baseGridPositionsRef.current[j];
            positions[j + 1] = baseGridPositionsRef.current[j + 1] + 
                            (Math.sin((ix + countRef.current) * 0.3) * 50) +
                            (Math.sin((iz + countRef.current) * 0.5) * 50);
            positions[j + 2] = baseGridPositionsRef.current[j + 2];
            
            // 为粒子添加缩放动画效果
            const baseScaleFactor = 1.0;
            const waveScaleFactor = (Math.sin((ix + countRef.current) * 0.3 + 1) * 0.2 +
                                  Math.sin((iz + countRef.current) * 0.5 + 1) * 0.2);
                                  
            scales[i] = baseScaleFactor + waveScaleFactor;
            
            i++;
            j += 3;
          }
        }
        
        needsUpdate = true;
      }
    }
    
    if (needsUpdate) {
      geometry.attributes.position.needsUpdate = true;
      geometry.attributes.scale.needsUpdate = true;
    }
    
    // 动画完成时触发回调，避免重复触发
    const now = Date.now();
    const minStateDuration = 300; // 状态持续的最小时间，防止频繁触发回调
    
    if (
      (animationState === 'targetReached' && prevAnimationStateRef.current === 'targetReached' && !animationCompletedRef.current['targetReached'] && 
       now - stateChangeTimeRef.current > minStateDuration) ||
      (animationState === 'initial' && prevAnimationStateRef.current === 'initial' && !animationCompletedRef.current['initial'] && 
       now - stateChangeTimeRef.current > minStateDuration)
    ) {
      // 在设置动画完成标志之前先触发回调，避免标志阻止回调
      stateChangeLockedRef.current = false; // 确保状态锁被释放
      animationCompletedRef.current[animationState] = true;
      
      // 使用setTimeout防止状态更新过快
      setTimeout(() => {
        onAnimationComplete?.(animationState);
      }, 50);
    }
  }, [animationState, onAnimationComplete, isClient])
  
  return {
    scene: sceneRef.current,
    camera: cameraRef.current,
    particles: particlesRef.current,
    updateParticles,
    animationState,
    setAnimationState: setAnimationStateWithLog
  }
} 