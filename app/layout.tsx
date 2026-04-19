import type { ReactNode } from "react";
import "./globals.css";

export const metadata = {
  title: "Gulf Watch",
  description: "Hurricane rapid-intensification prediction dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
