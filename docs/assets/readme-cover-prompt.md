# Ảnh cover README

- Tệp: `readme-cover.png`.
- Tạo bằng công cụ tạo ảnh AI tích hợp trong Codex, ngày 16/11/2026.
- Tham chiếu phong cách: [giao diện Hugging Face hiện tại của dự án](https://huggingface.co/spaces/AnomalGreatad/wood-grain-classification), được xem trực tiếp khi tạo ảnh.
- Ảnh vân gỗ là minh họa do AI tạo, không phải mẫu từ dataset, ảnh Grad-CAM hoặc kết quả suy luận thực tế.
- Bản hiện tại đã bỏ các khối biểu tượng, đường nối, điểm tròn và góc quét; chỉ giữ ảnh vân gỗ trong khung trắng ở phần minh họa bên phải, theo yêu cầu của người dùng.
- Nội dung dùng Nhóm 18 theo thông tin nhóm đã xác nhận.

## Prompt tạo bản gốc

```text
Use case: ads-marketing.
Asset type: a finished GitHub README cover banner for the WoodGrain-AI university wood-species classification project.
Create one beautiful wide panoramic banner, approximately 3:1 aspect ratio (1800 x 600 preferred), optimized for full-width display in a GitHub README. All typography must remain readable when the image is displayed at 900 pixels wide.
Reference image: the most recent screenshot of the project's actual Hugging Face web app. Use it ONLY as a style reference: pale icy cyan on the left gently blending into white and very pale lavender on the right; royal blue accents, deep navy typography, generous breathing room, elegant rounded white cards, extremely soft blue shadows. Do not copy the browser chrome, Hugging Face navigation, tabs, upload button, or any screenshots of text. Create a custom composed cover, not a website screenshot.
Composition: restrained professional product identity, left 55 percent for a large typographic title and concise Vietnamese subtitle, right 45 percent for a polished wood-grain recognition illustration. The right illustration is one macro photograph-like square of natural pale tan wood with fine visible grain and subtle pore patterns inside a floating rounded white card, with a thin translucent blue scanning frame and a few small blue feature points linked to a tiny layered neural-network symbol. Integrate two restrained floating white chips, without numeric predictions. This is a conceptual cover illustration, not a real inference result. Maintain a clean, carefully aligned layout and ample safe margins on every side.
Render only these exact text strings, spelled accurately with correct Vietnamese diacritics:
Main title, very large bold modern sans-serif: "WoodGrain-AI"
Subtitle, clear medium size: "Nhận diện loài gỗ từ ảnh vân gỗ"
Three small tasteful blue-on-white pill labels below the subtitle: "ResNet18" and "12 loài gỗ" and "WOOD-AUTH"
Small footer line: "INT3401 · Nhóm 18"
Use sophisticated sans-serif typography matching the screenshot, heavy title and calm readable supporting text. Make the title the visual anchor. No performance statistics, no unverified accuracy percentages, no four-model claims, no fake predictions, no extra text, no stock watermarks, no robots, no people, no trees or forest scenery, no busy circuitry, no dark background. Crisp polished design, subtle visual depth, natural wood texture balanced with minimal scientific blue scan graphics. Opaque background.
```

## Prompt chỉnh sửa bản hiện tại

```text
Edit the provided existing README cover banner. Target image is the clean project cover supplied by local file, not the red annotated screenshot.
Make exactly this simplification on the RIGHT side: remove ALL three small floating white icon cards to the right of the wood photograph (the layers icon card, the neural-network node card, and the magnifying-glass card). Remove ALL connecting blue curves/lines and ALL small circular feature points, including the portions that overlay the wood photograph. Remove the four luminous blue corner scan brackets too, so the right-hand illustration consists ONLY of the natural wood-grain photograph in its simple rounded white frame and its existing soft shadow, with no graphics on the wood.
Preserve the natural wood texture and rectangular rounded white photo card. Shift this single wood card slightly rightwards so it is visually centered in the freed right-hand region, leaving generous margins and a balanced banner.
Fill removed areas seamlessly with the same very pale cyan-white-lavender gradient background. Preserve the left side, all existing text, blue title icon, three text pills, typography, colors, scale, footer and overall banner aspect ratio. Preserve these exact strings and Vietnamese diacritics: "WoodGrain-AI"; "Nhận diện loài gỗ từ ảnh vân gỗ"; "ResNet18"; "12 loài gỗ"; "WOOD-AUTH"; "INT3401 · Nhóm 18".
This is a precise cleanup of the existing artwork, not a redesign. No new symbols, no nodes, no connector lines, no extra decoration, no new text. Keep the opaque background.
```
