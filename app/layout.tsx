import type { ReactNode } from "react";

export const metadata = {
  title: "Gulf Watch",
  description: "Hurricane rapid-intensification prediction dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "system-ui, sans-serif", background: "#0b0d10", color: "#f0ede8" }}>
        {children}
      </body>
    </html>
  );
}
