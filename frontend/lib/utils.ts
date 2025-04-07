import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import * as THREE from "three"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// 创建柔和的圆形纹理
export function createCircleTextureSoft(size: number = 64): THREE.CanvasTexture | THREE.Texture {
  // 服务器端渲染检查
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    // 返回一个空纹理，避免在SSR期间出错
    return new THREE.Texture();
  }
  
  const canvas = document.createElement('canvas')
  const context = canvas.getContext('2d')
  
  canvas.width = size
  canvas.height = size

  if (!context) {
    throw new Error('Failed to get canvas context')
  }

  // 透明背景
  context.clearRect(0, 0, size, size)

  // 径向渐变
  const gradient = context.createRadialGradient(
    size / 2, size / 2, 0,
    size / 2, size / 2, size / 2
  )
  gradient.addColorStop(0, 'rgba(255,255,255,1)')
  gradient.addColorStop(0.6, 'rgba(255,255,255,0.9)')
  gradient.addColorStop(1, 'rgba(255,255,255,0)')

  // 填充渐变
  context.fillStyle = gradient
  context.fillRect(0, 0, size, size)

  // 创建纹理并禁用 FLIP_Y 和 PREMULTIPLY_ALPHA 设置
  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  // 解决WebGL警告：禁用这些在3D纹理上不允许的设置
  texture.flipY = false
  texture.premultiplyAlpha = false
  
  return texture
}
