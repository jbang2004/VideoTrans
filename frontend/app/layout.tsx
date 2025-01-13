import "@/styles/globals.css"

export const metadata = {
  title: "Video Translation",
  description: "Upload and translate your videos",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

