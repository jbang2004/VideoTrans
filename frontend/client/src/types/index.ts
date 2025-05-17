export interface Subtitle {
  id: string;
  startTime: string;
  endTime: string;
  text: string;
  translation: string;
  isEditing?: boolean;
  isPanelClosed?: boolean; // This field was in the parent component's original interface, including it for completeness, can be refined if not used by all consumers.
} 