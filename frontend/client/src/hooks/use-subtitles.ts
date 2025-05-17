import { useState, useEffect, useRef } from 'react';
import { Subtitle } from '@/types';
import { supabase } from '@/lib/supabaseClient'; // Ensure this path is correct

// Helper function to convert milliseconds to HH:MM:SS string
const formatTime = (ms: number): string => {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
};

interface UseSubtitlesProps {
  loadCondition: boolean; 
  // We might not need loadCondition anymore if fetching is triggered by a button
  // Or it could be used to show the panel initially if some conditions are met
  // For now, I'll keep it but the primary loading will be via fetchSubtitles
  initialTaskId?: string; // Optional: if a task_id is known on load
}

export function useSubtitles({ loadCondition, initialTaskId }: UseSubtitlesProps) {
  const [subtitles, setSubtitles] = useState<Subtitle[]>([]);
  const [showSubtitles, setShowSubtitles] = useState<boolean>(false);
  const [editingSubtitleId, setEditingSubtitleId] = useState<string | null>(null);
  const [isLoadingSubtitles, setIsLoadingSubtitles] = useState<boolean>(false);
  const [subtitleError, setSubtitleError] = useState<string | null>(null);
  const panelClosedByUserRef = useRef<boolean>(false);

  const fetchSubtitles = async (taskId: string, targetLanguageCode: string) => {
    if (!taskId) {
      setSubtitleError("Task ID is required to fetch subtitles.");
      return;
    }
    setIsLoadingSubtitles(true);
    setSubtitleError(null);
    setSubtitles([]); // Clear previous subtitles

    try {
      const { data, error } = await supabase
        .from('sentences')
        .select('id, sentence_index, raw_text, start_ms, end_ms, trans_text')
        .eq('task_id', taskId)
        // .eq('language', targetLanguageCode) // Assuming 'trans_text' is already for the target lang OR we'd need a lang column
                                          // If trans_text is always the target language of the task, no need for this.
                                          // If 'sentences' stores multiple translations, we'd need a language column.
                                          // Based on schema, 'trans_text' seems singular.
        .order('sentence_index', { ascending: true });

      if (error) {
        console.error("Error fetching subtitles:", error);
        setSubtitleError(`Failed to fetch subtitles: ${error.message}`);
        setShowSubtitles(false);
        return;
      }

      if (data) {
        const formattedSubtitles: Subtitle[] = data.map(s => ({
          id: String(s.id),
          startTime: formatTime(s.start_ms),
          endTime: formatTime(s.end_ms),
          text: s.raw_text || "", // Original text
          translation: s.trans_text || "", // Translated text for the task's target language
        }));
        setSubtitles(formattedSubtitles);
        setShowSubtitles(true); // Show panel once data is fetched
        panelClosedByUserRef.current = false; // Reset closed by user flag
      } else {
        setSubtitles([]);
        setShowSubtitles(true); // Show panel but it will be empty
        setSubtitleError("No subtitles found for this task.");
      }
    } catch (err: any) {
      console.error("Client-side error fetching subtitles:", err);
      setSubtitleError(`An unexpected error occurred: ${err.message}`);
      setShowSubtitles(false);
    } finally {
      setIsLoadingSubtitles(false);
    }
  };
  
  // Optional: Load initial subtitles if taskId is provided and conditions met
  useEffect(() => {
    if (loadCondition && initialTaskId && !showSubtitles && subtitles.length === 0 && !panelClosedByUserRef.current) {
      // For now, this won't auto-fetch. Fetching is manual via the button.
      // If auto-fetch on load is desired, call fetchSubtitles here with a default/known target language.
      // setShowSubtitles(true); // This would just show an empty panel if we don't fetch.
    }
  }, [loadCondition, initialTaskId, showSubtitles, subtitles.length]);


  const updateSubtitleTranslation = (id: string, newTranslation: string) => {
    setSubtitles(prevSubtitles =>
      prevSubtitles.map(sub =>
        sub.id === id ? { ...sub, translation: newTranslation } : sub
      )
    );
    // TODO: Add a call to update the translation in the database if needed
    // e.g., supabase.from('sentences').update({ trans_text: newTranslation }).eq('id', id);
  };

  const toggleEditMode = (id: string) => {
    setEditingSubtitleId(prevId => (prevId === id ? null : id));
  };

  const closeSubtitlesPanel = () => {
    setShowSubtitles(false);
    panelClosedByUserRef.current = true;
  };
  
  const resetSubtitlesState = () => { // Renamed to avoid conflict if used elsewhere
    setSubtitles([]);
    setShowSubtitles(false);
    setEditingSubtitleId(null);
    setIsLoadingSubtitles(false);
    setSubtitleError(null);
    panelClosedByUserRef.current = false;
  }

  return {
    subtitles,
    // setSubtitles, // Not exposing directly anymore
    showSubtitles,
    // setShowSubtitles, // Not exposing directly
    editingSubtitleId,
    isLoadingSubtitles,
    subtitleError,
    fetchSubtitles, // Expose the new fetch function
    updateSubtitleTranslation,
    toggleEditMode,
    closeSubtitlesPanel,
    resetSubtitlesState, // Expose the reset function
    // panelClosedByUserRef // Internal ref, probably not needed outside
  };
} 