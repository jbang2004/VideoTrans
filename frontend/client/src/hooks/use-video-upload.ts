import { useState } from 'react';
import { supabase } from '@/lib/supabaseClient';
import { User } from '@supabase/supabase-js';
import * as tus from 'tus-js-client';

interface VideoUploadState {
  isUploading: boolean;
  uploadProgress: number;
  uploadComplete: boolean;
  videoPreviewUrl: string | null;
  uploadError: Error | null;
}

interface VideoUploadActions {
  initiateUpload: (fileToUpload: File, user: User, token: string) => Promise<void>;
  resetUploadState: () => void;
}

const SUPABASE_PROJECT_ID = "aupsrnasyirsqrjtjvok"; // Store this in env vars ideally

export function useVideoUpload(): VideoUploadState & VideoUploadActions {
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploadComplete, setUploadComplete] = useState<boolean>(false);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<Error | null>(null);

  const resetUploadState = () => {
    setIsUploading(false);
    setUploadProgress(0);
    setUploadComplete(false);
    setVideoPreviewUrl(null);
    setUploadError(null);
  };

  const initiateUpload = async (fileToUpload: File, user: User, token: string) => {
    resetUploadState();
    setIsUploading(true);

    const fileExt = fileToUpload.name.split('.').pop();
    const objectName = `${user.id}_${Date.now()}.${fileExt}`;
    const bucketName = 'videos';
    const tusEndpoint = `https://${SUPABASE_PROJECT_ID}.supabase.co/storage/v1/upload/resumable`;

    const upload = new tus.Upload(fileToUpload, {
      endpoint: tusEndpoint,
      retryDelays: [0, 3000, 5000, 10000, 20000],
      headers: {
        authorization: `Bearer ${token}`,
        'x-upsert': 'true',
      },
      metadata: {
        bucketName: bucketName,
        objectName: objectName,
        contentType: fileToUpload.type,
        cacheControl: '3600',
      },
      chunkSize: 6 * 1024 * 1024,
      onError: (error) => {
        console.error("TUS Upload Failed:", error);
        setUploadError(error);
        setIsUploading(false);
        setUploadProgress(0);
      },
      onProgress: (bytesUploaded, bytesTotal) => {
        const percentage = Math.round((bytesUploaded / bytesTotal) * 100);
        setUploadProgress(percentage);
      },
      onSuccess: async () => {
        console.log("TUS Upload Succeeded for:", fileToUpload.name, upload.url);
        const finalObjectNameInBucket = objectName;

        const { data: publicUrlData } = supabase.storage
          .from(bucketName)
          .getPublicUrl(finalObjectNameInBucket);

        if (!publicUrlData || !publicUrlData.publicUrl) {
          const err = new Error("无法获取视频的公开URL。");
          console.error(err);
          setUploadError(err);
          setIsUploading(false);
          return;
        }
        setVideoPreviewUrl(publicUrlData.publicUrl);

        const videoDataToInsert = {
          user_id: user.id,
          file_name: fileToUpload.name,
          storage_path: finalObjectNameInBucket,
          status: 'uploaded',
        };

        const { error: dbError } = await supabase
          .from('videos')
          .insert([videoDataToInsert]);

        if (dbError) {
          console.error("数据库插入失败:", dbError);
          setUploadError(new Error(dbError.message));
          setIsUploading(false); // Keep URL for preview but flag DB error
          return;
        }

        setIsUploading(false);
        setUploadComplete(true);
      },
    });

    upload.start();
  };

  return {
    isUploading,
    uploadProgress,
    uploadComplete,
    videoPreviewUrl,
    uploadError,
    initiateUpload,
    resetUploadState,
  };
} 