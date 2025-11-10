const UploadButton = ({ onClick, disable, children }) => {
  return (
    <>
      <button
        className="px-3 py-2 px-md-4 button-bg border-0 shadow rounded-2 fw-medium text-white"
        onClick={onClick}
        disabled={disable}
      >
        {children}
      </button>
    </>
  );
};

export default UploadButton;
