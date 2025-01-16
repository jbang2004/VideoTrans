import { useRef, useState, useEffect } from 'react'
import type { PlayerState, PlayerControls } from '../types'

export function useVideoPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [state, setState] = useState<PlayerState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    volume: 1,
    localVideoUrl: null
  })

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleTimeUpdate = () => {
      setState(prev => ({
        ...prev,
        currentTime: video.currentTime,
        duration: video.duration || prev.duration
      }))
    }

    const handleLoadedMetadata = () => {
      setState(prev => ({
        ...prev,
        duration: video.duration
      }))
    }

    const handleVolumeChange = () => {
      setState(prev => ({
        ...prev,
        volume: video.volume
      }))
    }

    const handlePlay = () => {
      setState(prev => ({
        ...prev,
        isPlaying: true
      }))
    }

    const handlePause = () => {
      setState(prev => ({
        ...prev,
        isPlaying: false
      }))
    }

    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('volumechange', handleVolumeChange)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('volumechange', handleVolumeChange)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
    }
  }, [])

  const controls: PlayerControls = {
    play: () => {
      if (videoRef.current) {
        videoRef.current.play()
      }
    },
    pause: () => {
      if (videoRef.current) {
        videoRef.current.pause()
      }
    },
    seek: (time: number) => {
      if (videoRef.current) {
        videoRef.current.currentTime = time
      }
    },
    setVolume: (volume: number) => {
      if (videoRef.current) {
        videoRef.current.volume = volume
      }
    },
    setLocalVideoUrl: (url: string | null) => {
      setState(prev => ({ ...prev, localVideoUrl: url }))
      if (videoRef.current && url) {
        videoRef.current.src = url
      }
    }
  }

  return {
    videoRef,
    state,
    controls
  }
} 