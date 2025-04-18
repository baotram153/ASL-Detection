// import React, { useEffect, useRef } from 'react';

// function CameraFeed() {
//   // 1) A ref to the <video> element in our JSX
//   const videoRef = useRef(null);

//   useEffect(() => {
//     let localStream = null;

//     // 2) Function to request user media and set up the video element
//     async function startCamera() {
//       try {
//         // Request camera access (video: true). Add audio if needed: { video: true, audio: true }
//         const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//         localStream = stream;

//         // Attach the stream to the <video> element
//         if (videoRef.current) {
//           videoRef.current.srcObject = stream;
//         }
//       } catch (err) {
//         console.error('Error accessing camera:', err);
//       }
//     }

//     // 3) Start the camera on mount
//     startCamera();

//     // 4) Cleanup: stop the camera feed on unmount
//     return () => {
//       if (localStream) {
//         localStream.getTracks().forEach((track) => {
//           track.stop();
//         });
//       }
//     };
//   }, []);

//   return (
//     <div>
//       <h2>Camera Feed</h2>
//       <video
//         ref={videoRef}
//         autoPlay
//         playsInline
//         style={{ width: '400px', backgroundColor: '#000' }}
//       />
//     </div>
//   );
// }

// export default CameraFeed;