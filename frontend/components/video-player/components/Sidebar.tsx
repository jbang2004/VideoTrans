import React from 'react'
import { Button } from '../../ui/button'
import { ScrollArea } from '../../ui/scroll-area'
import { Clock, Folder, Heart, Library, Star, Upload, Video } from 'lucide-react'

export function Sidebar() {
  return (
    <div className="w-[240px] border-r border-white/10 bg-white/[0.02] backdrop-blur-sm">
      <ScrollArea className="h-full">
        <div className="p-4">
          <div className="mb-8">
            <h1 className="mb-4 text-xl font-bold bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent">Sora</h1>
          </div>

          <nav className="space-y-6">
            <div>
              <h2 className="mb-2 text-sm font-semibold text-white/50">Explore</h2>
              <div className="space-y-1">
                <Button variant="ghost" className="w-full justify-start gap-2 text-white/70 hover:text-white hover:bg-white/10">
                  <Clock className="h-4 w-4" />
                  Recent
                </Button>
                <Button variant="ghost" className="w-full justify-start gap-2 text-white/70 hover:text-white hover:bg-white/10">
                  <Star className="h-4 w-4" />
                  Featured
                </Button>
                <Button variant="ghost" className="w-full justify-start gap-2 text-white/70 hover:text-white hover:bg-white/10">
                  <Heart className="h-4 w-4" />
                  Saved
                </Button>
              </div>
            </div>

            <div>
              <h2 className="mb-2 text-sm font-semibold text-white/50">Library</h2>
              <div className="space-y-1">
                <Button variant="ghost" className="w-full justify-start gap-2 text-white/70 hover:text-white hover:bg-white/10">
                  <Video className="h-4 w-4" />
                  All videos
                </Button>
                <Button variant="ghost" className="w-full justify-start gap-2 text-white/70 hover:text-white hover:bg-white/10">
                  <Star className="h-4 w-4" />
                  Favorites
                </Button>
                <Button variant="ghost" className="w-full justify-start gap-2 text-white/70 hover:text-white hover:bg-white/10">
                  <Upload className="h-4 w-4" />
                  Uploads
                </Button>
                <Button variant="ghost" className="w-full justify-start gap-2 text-white/70 hover:text-white hover:bg-white/10">
                  <Folder className="h-4 w-4" />
                  New folder
                </Button>
              </div>
            </div>
          </nav>
        </div>
      </ScrollArea>
    </div>
  )
} 