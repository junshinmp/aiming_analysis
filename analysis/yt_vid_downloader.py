import yt_dlp
import os
from pathlib import Path

def download_video(url, output_folder="training_data"):
    output_folder = Path(output_folder)

    print(f"Youtube videos will be put into the {output_folder} directory")
    os.makedirs(output_folder, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/best[vcodec^=avc1]',
        'outtmpl': str(output_folder / '%(title)s.%(ext)s'),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        final_path_string = ydl.prepare_filename(info_dict)
        ydl.download([url])

    return str(Path(final_path_string))

if __name__ == '__main__':
    url = 'https://www.youtube.com/watch?v=d-RGJPvFuNE'
    video_path = download_video(url)
    print(f"Video saved at {str(video_path)}")