import { test, vi, expect } from "vitest";
import { screen, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import '@testing-library/jest-dom'

// Components
import App from "./App";
import UploadButton from "./components/UploadButton";

test("Check if the file input component exist in the project", () => {
    render(<App/>);
    const fileInputeElement = screen.getByPlaceholderText("SCC File Input");
    expect(fileInputeElement).toBeInTheDocument();
});

test("Check if the button component is clickable", async () => {
    const handleClick = vi.fn(); // Similar to jest.fn()
    render(<UploadButton onClick={handleClick} >w/Normalization</UploadButton>);
    const buttonElement = screen.getByRole("button", { name: "w/Normalization" });
    await userEvent.click(buttonElement);
    expect(handleClick).toHaveBeenCalledTimes(1);
})
