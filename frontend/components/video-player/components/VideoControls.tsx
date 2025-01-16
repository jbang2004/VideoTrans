import React from 'react'
import { Button } from '../../ui/button'
import { Slider } from '../../ui/slider'
import { Volume2 } from 'lucide-react'
import { cn } from '../../../lib/utils'
import { formatTime } from '../utils/format'
import type { PlayerState, PlayerControls } from '../types'

interface VideoControlsProps {
  state: PlayerState
  controls: PlayerControls
}

export function VideoControls({ state, controls }: VideoControlsProps) {
  const { currentTime, duration, volume } = state
  const { seek, setVolume } = controls

  const handleSeek = (value: number[]) => {
    seek(value[0])
  }

  const handleVolumeChange = (value: number[]) => {
    setVolume(value[0])
  }

  return (
    <div className="w-full flex items-center gap-2">
      <span className="text-xs text-white/70 min-w-[40px]">{formatTime(currentTime)}</span>
      <div className="relative w-full h-1 group">
        <div 
          className="absolute inset-0 bg-white/20 rounded-full"
        />
        <div 
          className="absolute inset-y-0 left-0 bg-white/80 rounded-full transition-all"
          style={{ width: `${(currentTime / duration) * 100}%` }}
        />
        <Slider
          value={[currentTime]}
          max={duration}
          step={0.1}
          className="absolute inset-0 appearance-none bg-transparent [&>span]:opacity-0 group-hover:[&>span]:opacity-100 [&>span]:transition-opacity [&>span]:duration-200"
          onValueChange={handleSeek}
        />
      </div>
      <span className="text-xs text-white/70 min-w-[40px]">{formatTime(duration)}</span>

      <div className="flex items-center gap-2">
        <Button 
          size="icon" 
          variant="ghost" 
          className="hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white"
        >
          <Volume2 className="h-4 w-4" />
        </Button>
        <div className="relative w-[60px] h-1 group">
          <div 
            className="absolute inset-0 bg-white/20 rounded-full"
          />
          <div 
            className="absolute inset-y-0 left-0 bg-white/80 rounded-full transition-all"
            style={{ width: `${volume * 100}%` }}
          />
          <Slider
            value={[volume]}
            max={1}
            step={0.01}
            className="absolute inset-0 appearance-none bg-transparent [&>span]:opacity-0 group-hover:[&>span]:opacity-100 [&>span]:transition-opacity [&>span]:duration-200"
            onValueChange={handleVolumeChange}
          />
        </div>
      </div>
    </div>
  )
} 