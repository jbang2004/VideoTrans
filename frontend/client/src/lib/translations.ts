export type Language = 'zh' | 'en'; // Add other languages as needed

export interface Translations {
  title: string;
  description: string;
  uploadVideoLabel: string;
  selectLanguageLabel: string;
  startPreprocessingLabel: string;
  startTranslationLabel: string;
  uploadingLabel: string;
  translatingLabel: string;
  processingLabel: string;
  processingSubtitlesLabel: string;
  playLabel: string;
  pauseLabel: string;
  editLabel: string;
  saveLabel: string;
  generateLabel: string;
  originalSubtitleLabel: string;
  translatedSubtitleLabel: string;
  jumpToLabel: string;
  closeLabel: string;
  languageLabel: string;
  translateButtonLabel: string;
  loadingLabel: string;
  loadingSubtitlesLabel: string;
  errorLabel: string;
  retryLabel: string;
  noSubtitlesFoundLabel: string;
  alertMessages: {
    selectVideoFirst: string;
    userInfoIncomplete: string;
    userInfoRefreshFailed: string;
    sessionTokenInvalid: string;
    authTokenFailed: string;
    uploadFailed: (message: string) => string;
    dbError: (message: string) => string;
    subtitlesTranslated: (language: string) => string;
    generatingVideo: string;
    uploadNotComplete: string;
  };
  languageOptions: Array<{ value: string; label: string }>;
}

const translations: Record<Language, Translations> = {
  en: {
    title: "Video Translation",
    description: "Upload videos, automatically extract subtitles and translate into multiple languages. Support batch export and editing.",
    uploadVideoLabel: "Upload",
    selectLanguageLabel: "Select language",
    startPreprocessingLabel: "Start Preprocessing",
    startTranslationLabel: "Translate",
    uploadingLabel: "Uploading...",
    translatingLabel: "Translating...",
    processingLabel: "Processing...",
    processingSubtitlesLabel: "Extracting subtitles...",
    playLabel: "Play",
    pauseLabel: "Pause",
    editLabel: "Edit",
    saveLabel: "Save",
    generateLabel: "Start Generating",
    originalSubtitleLabel: "Original Subtitle",
    translatedSubtitleLabel: "Translated Subtitle",
    jumpToLabel: "Jump to",
    closeLabel: "Close",
    languageLabel: "Language",
    translateButtonLabel: "Translate Subtitles",
    loadingLabel: "Loading...",
    loadingSubtitlesLabel: "Loading subtitles...",
    errorLabel: "Error",
    retryLabel: "Try Again",
    noSubtitlesFoundLabel: "No subtitles found. Select language and click Translate.",
    alertMessages: {
      selectVideoFirst: "Please select a video file first.",
      userInfoIncomplete: "User info incomplete, attempting to refresh...",
      userInfoRefreshFailed: "Could not retrieve valid user information or session. Please refresh the page or log in again and try.",
      sessionTokenInvalid: "User information or session token is invalid, please try again.",
      authTokenFailed: "Failed to get authorization token, please try again.",
      uploadFailed: (message: string) => `Upload failed: ${message}`,
      dbError: (message: string) => `Database error: ${message}`,
      subtitlesTranslated: (language: string) => `Subtitles translated to ${language}`,
      generatingVideo: "Starting to generate translated video!",
      uploadNotComplete: "Video upload is not yet complete. Please wait.",
    },
    languageOptions: [
      { value: "zh", label: "Chinese" },
      { value: "en", label: "English" },
      { value: "ja", label: "Japanese" },
      { value: "ko", label: "Korean" },
      { value: "fr", label: "French" },
      { value: "de", label: "German" },
      { value: "es", label: "Spanish" },
    ],
  },
  zh: {
    title: "视频翻译",
    description: "上传视频，自动提取字幕并翻译成多种语言。支持批量导出与编辑。",
    uploadVideoLabel: "上传视频",
    selectLanguageLabel: "选择语言",
    startPreprocessingLabel: "开始预处理",
    startTranslationLabel: "开始翻译",
    uploadingLabel: "上传中...",
    translatingLabel: "翻译中...",
    processingLabel: "处理中...",
    processingSubtitlesLabel: "正在提取字幕...",
    playLabel: "播放",
    pauseLabel: "暂停",
    editLabel: "编辑",
    saveLabel: "保存",
    generateLabel: "开始生成",
    originalSubtitleLabel: "原始字幕",
    translatedSubtitleLabel: "翻译字幕",
    jumpToLabel: "跳转",
    closeLabel: "关闭",
    languageLabel: "语言",
    translateButtonLabel: "翻译字幕",
    loadingLabel: "加载中...",
    loadingSubtitlesLabel: "字幕加载中...",
    errorLabel: "错误",
    retryLabel: "重试",
    noSubtitlesFoundLabel: "未找到字幕。请选择语言后点击翻译。",
    alertMessages: {
      selectVideoFirst: "请先选择一个视频文件。",
      userInfoIncomplete: "用户信息不完整，正在尝试重新获取...",
      userInfoRefreshFailed: "无法获取有效的用户信息或会话，请刷新页面或重新登录后尝试。",
      sessionTokenInvalid: "用户信息或会话令牌无效，请重试。",
      authTokenFailed: "无法获取授权令牌，请重试。",
      uploadFailed: (message: string) => `上传失败: ${message}`,
      dbError: (message: string) => `数据库错误: ${message}`,
      subtitlesTranslated: (language: string) => `字幕已翻译为${language}`,
      generatingVideo: "开始生成翻译后的视频！",
      uploadNotComplete: "视频尚未上传完成，请稍候。",
    },
    languageOptions: [
      { value: "zh", label: "中文" },
      { value: "en", label: "英文" },
      { value: "ja", label: "日语" },
      { value: "ko", label: "韩语" },
      { value: "fr", label: "法语" },
      { value: "de", label: "德语" },
      { value: "es", label: "西班牙语" },
    ],
  },
};

export const getTranslations = (language: Language): Translations => {
  return translations[language] || translations.en; // Fallback to English
}; 