export const formatTime = (seconds: number): string => {
  if (!seconds || isNaN(seconds)) return '00:00'
  
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.floor(seconds % 60)
  
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
}

// 根据环境使用不同的API地址
export const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '' // 生产环境下使用相对路径，通过Next.js的API路由
  : 'http://localhost:8000' // 开发环境使用本地服务器地址

export const LANGUAGES = [
  { label: '中文', value: 'zh' },
  { label: 'English', value: 'en' },
  { label: '日本語', value: 'ja' },
  { label: '한국어', value: 'ko' }
] as const

export const LANGUAGE_MAP = LANGUAGES.reduce((acc, { label, value }) => {
  acc[label] = value
  return acc
}, {} as Record<string, string>) 