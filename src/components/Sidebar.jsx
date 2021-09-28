import React from "react";
import { Layout, Menu, Divider, Row, Col } from "antd";
import {
  HomeFilled,
  ReadFilled,
  TwitterSquareFilled,
  LinkedinFilled,
  GithubFilled,
  BoxPlotOutlined,
} from "@ant-design/icons";
const { Sider } = Layout;
import { Avatar } from "antd";
import { Link } from "react-router-dom";
import profileImage from "../../static/profile_avatar.jpg";

const socialStyles = {
  fontSize: "2.5em",
  marginTop: "20px",
  marginBottom: "20px",
};

class Sidebar extends React.Component {
  render() {
    return (
      <Sider
        collapsed
        collapsedWidth={80}
        style={{
          overflow: "auto",
          height: "100vh",
          position: "fixed",
          left: 0,
        }}
      >
        <Link to={"/"}>
          <Avatar shape="circle" size={80} src={profileImage} />
        </Link>
        <hr />
        <Menu theme="dark" defaultSelectedKeys={["0"]} mode="inline">
          <Menu.Item>
            <HomeFilled />
            <span>{"Home"}</span>
            <Link to={"/"} />
          </Menu.Item>
          <Menu.Item>
            <ReadFilled />
            <span>{"About"}</span>
            <Link to={"/about"} />
          </Menu.Item>
          <Menu.Item>
            <BoxPlotOutlined />
            <span>{"Options Playground"}</span>
            <Link to={"/options"} />
          </Menu.Item>
          <Divider />
        </Menu>
        <Row
          style={{ paddingLeft: "25px", display: "flex", alignItems: "center" }}
        >
          <Col span={24}>
            <a href={"https://twitter.com/kapil_sh_"}>
              <TwitterSquareFilled style={socialStyles} />
            </a>
          </Col>
          <Col span={24}>
            <a href={"https://www.linkedin.com/in/sharma-k/"}>
              <LinkedinFilled style={socialStyles} />
            </a>
          </Col>
          <Col span={24}>
            <a href={"https://github.com/kapilsh"}>
              <GithubFilled style={socialStyles} />
            </a>
          </Col>
        </Row>
      </Sider>
    );
  }
}

export default Sidebar;
