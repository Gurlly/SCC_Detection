import Logo from "../assets/Logo.png";

const Header = () => {
  return (
    <>
      <header className="container-fluid p-3 p-md-4" id="header">
        <div className="container-md mx-md-auto p-0 d-flex justify-content-between align-items-center">
          <div className="d-flex align-items-center gap-3">
            <img src={Logo} alt="Application Logo" id="logo" />
            <h1 className="text-white m-0 fs-2">DS33-Thesis</h1>
          </div>
          <div className=" d-none d-md-flex align-items-center gap-4">
            <p className="m-0 fs-6 text-white fw-semibold">Cauba</p>
            <p className="m-0 fs-6 text-white fw-semibold">Calvo</p>
            <p className="m-0 fs-6 text-white fw-semibold">Martinez</p>
            <p className="m-0 fs-6 text-white fw-semibold">Tuazon</p>
          </div>
        </div>
      </header>
    </>
  );
};

export default Header;
