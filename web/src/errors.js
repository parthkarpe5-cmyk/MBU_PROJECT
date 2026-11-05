export function friendlyError(error, stage='camera') {
  if(stage==='model')return 'The face model could not load. Check your connection and try again. Camera access has been released.';
  if(stage==='expression')return 'The expression model could not load. You can keep using geometry monitoring or retry the model.';
  const messages={
    NotAllowedError:'Camera permission was denied. Allow camera access for this site in your browser, then try again.',
    NotFoundError:'No webcam was found. Connect a camera and try again.',
    NotReadableError:'The webcam is busy or unavailable. Close other camera apps and retry.',
    OverconstrainedError:'That camera is unavailable. Select the default camera and retry.',
  };
  return messages[error?.name] || 'Monitoring could not continue. Check your camera connection and try again.';
}
