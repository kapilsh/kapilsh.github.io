import React from "react";
import { Typography } from "antd";
import { CodeBlock, dracula } from "react-code-blocks";

const { Title, Paragraph } = Typography;

class LearningCuda extends React.Component {
    render() {
        return (
            <>
                <Typography>
                    <Paragraph>
                        Recently, I have deep-diving into cuda and ML compilation and wanted to share my journey of learning CUDA in the last couple of months.
                    </Paragraph>
                    <Title level={3}>CUDA Mode</Title>
                    <Title level={3}>Writing my first CUDA kernel</Title>
                    <Title level={3}>Loading the kernel in python</Title>
                    <Title level={3}>Profilers</Title>
                    <Title level={3}>Conclusion</Title>
                </Typography>
            </>
        );
    }
}

export default LearningCuda;